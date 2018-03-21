import datetime
import multiprocessing as mp
import os
import random
import time
from functools import partial

import dropbox
import IPython
import numpy as np
from dropbox.exceptions import ApiError, DropboxException, InternalServerError
from dropbox.files import WriteMode

from utils.filesystem import longest_common_suffix


def get_dropbox_client(token_file):
    """Returns a Dropbox client from a given token file.
    """
    with open(token_file) as f:
        token = f.readline()[:-1]
    return dropbox.Dropbox(token)


def get_writemode(mode):
    """
    :ivar add: Do not overwrite an existing file if there is a conflict. The
        autorename strategy is to append a number to the file name. For example,
        "document.txt" might become "document (2).txt".
    :ivar overwrite: Always overwrite the existing file. The autorename strategy
        is the same as it is for ``add``.
    :ivar str update: Overwrite if the given "rev" matches the existing file's
        "rev". The autorename strategy is to append the string "conflicted copy"
        to the file name. For example, "document.txt" might become "document
        (conflicted copy).txt" or "document (Panda's conflicted copy).txt".
    """
    assert mode in ['add', 'overwrite', 'update']
    return WriteMode(mode)


def get_modified_time(dbx, dbx_file):
    """Returns the time at which the `dbx_file` was last modified on server side.

    If no meta data was available (file not existing), `None` is returned.
    """
    try:
        mtime = dbx.files_get_metadata(dbx_file).server_modified.timestamp()
    except ApiError:
        mtime = None
    return mtime


def retry(function, *args, max_retries=10, wait=1, **kwargs):
    """Repeats a function call with `args` and `kwargs` a number of times, breaking
    if it returns successfully.
    """
    done = False
    i = 0
    while not done and i < max_retries:
        time.sleep(wait)
        try:
            out = function(*args, **kwargs)
            done = True
        except:
            pass
        i += 1
    info = "Succeded after " + str(i) + " retries." if done else "Failed after " + str(i) + " retries."
    return {'out': out, 'info': info}


def compare_modified_times(dbx, src_files, dbx_files):
    """Compares the times local files and corresponding dropbox files were modified.

    Return a dictionary for local files and server files indicating if each file (key) is newest 
    on the local side or server side with a True 
    """
    # Get dropbox files meta information all files in all unique parenting folders
    # (This yields at max as many requests as one request per file would have made)
    parent_folders = {os.path.split(dbx_f)[0] for dbx_f in dbx_files}
    all_metas = []
    for p_folder in parent_folders:
        all_metas.extend(dbx.files_list_folder(p_folder).entries)
    f_metas = [f_meta for f_meta in all_metas if type(f_meta) is dropbox.files.FileMetadata]
    # Files in metas not in `dbx_files` are removed.
    f_metas = [f_meta for f_meta in f_metas if f_meta.path_display in dbx_files]
    # Files in `dbx_files` that are not in metas are added (as modified at epoch)
    f_metas_files = [f_meta.path_display for f_meta in f_metas]
    for dbx_f in dbx_files:
        if dbx_f not in f_metas_files:
            this_missing_file_meta = dropbox.files.FileMetadata(path_display=dbx_f, server_modified=datetime.datetime.utcfromtimestamp(0))
            f_metas.append(this_missing_file_meta)
    # Sort to get same order of files in metas as in dbx_files
    f_metas_sorted = [None]*len(dbx_files)
    for i, dbx_f in enumerate(dbx_files):
        j = 0
        for f_meta in f_metas:
            if dbx_f == f_meta.path_display:
                break
            j += 1
        f_metas_sorted[i] = f_metas[j]
    f_metas = f_metas_sorted
    dbx_file_time = {f_meta.path_display: f_meta.server_modified.timestamp() for f_meta in f_metas}
    # Get modified times for local
    src_file_time = {}
    for src_f in src_files:
        if os.path.exists(src_f):
            src_file_time[src_f] = os.path.getmtime(src_f)
        else:
            src_file_time[src_f] = 0
    # Compare all pairs of files
    for src_f, dbx_f in zip(src_files, dbx_files):
        if src_file_time[src_f] == dbx_file_time[dbx_f]:
            # Both sides have equal mod times
            src_file_time[src_f] = False
            dbx_file_time[dbx_f] = False
        else:
            # One of the sides has newer version
            local_is_newest = src_file_time[src_f] > dbx_file_time[dbx_f]
            src_file_time[src_f] = local_is_newest
            dbx_file_time[dbx_f] = not local_is_newest
    return src_file_time, dbx_file_time


def upload_directory(dbx, src_dir, dbx_dir, upload_older_files=True, do_parallel=True, mode='overwrite'):
    """Copy an entire directory including all files and folders to Dropbox
    """
    print("Uploading directory to dropbox...")
    print("    Source: " + src_dir)
    print("    Target: " + dbx_dir)
    for src_root, src_dirs, src_files in os.walk(src_dir):
        # Create all directories in this directory
        src_dirs = [os.path.join(dbx_dir, d) for d in src_dirs]
        create_directories_dbx(dbx, src_dirs)
        # Upload all files in this directory
        src_files = [os.path.join(src_root, f) for f in src_files]
        dbx_files = [os.path.join(dbx_dir, os.path.relpath(f, src_dir)) for f in src_files]
        upload_files(dbx, src_files, dbx_files, upload_older_files=upload_older_files, do_parallel=do_parallel, mode=mode)
    print("    Uploading done")


def upload_file(dbx, src_file, dbx_file, upload_older_file=True, max_retries=10, mode='overwrite'):
    """Uploads a single `src_file` to the corresponding `dbx_file` in dropbox.
    """
    assert os.path.exists(src_file)
    mode = get_writemode(mode)
    # Compare the server file time stamp with local
    if not upload_older_file:
        print("Upload file")
        m_time_local = os.path.getmtime(src_file)
        m_time_server = get_modified_time(dbx, dbx_file)
        # If the file exists on server and has newer modification time
        if m_time_server is not None and m_time_local < m_time_server:
            return
    # Upload the local file
    with open(src_file, 'rb') as f:
        try:
            dbx.files_upload(f.read(), dbx_file, mode=mode)
            print("    Uploaded: " + dbx_file, end='\n')
        except ApiError as err:
            m = "    Failed to upload " + src_file + ": "
            if err.error.is_path() and err.error.get_path().is_insufficient_space():
                m += 'Insufficient space.'
            elif err.error.is_path() and err.error.get_path().is_conflict():
                if err.error.get_path().get_conflict().is_file():
                    # File already exists and mode is not overwrite
                    m += "Write conflict on file."
            print(m)
        except InternalServerError as err:
            m = "    Failed to upload " + src_file + ": Internal server error."
            retry(dbx.files_upload, f.read(), dbx_file, mode=mode, max_retries=10, wait=0.5)


def upload_files(dbx, src_files, dbx_files, upload_older_files=True, do_parallel=True, max_retries=10, mode='overwrite'):
    """Uploads a number of `src_files` to the corresponding `dbx_files` paths.

    The source files must exist while the dbx files may or may not exist. If they don't they will be created, if they do, 
    the upload will follow the `mode`.
    """
    assert len(src_files) == len(dbx_files)
    if not src_files:
        return
    # Force upload older files on per file basis, since here, check is made on batch if required, which is faster.
    if not upload_older_files:
        # Keep only files that are newest on source side
        src_newest, dbx_newest = compare_modified_times(dbx, src_files, dbx_files)
        src_files, dbx_files = [], []
        for src_f, upload_src, dbx_f, _ in zip(*zip(*sorted(src_newest.items())), *zip(*sorted(dbx_newest.items()))):
            if upload_src:
                src_files.append(src_f)
                dbx_files.append(dbx_f)
        if not src_files:
            return
    kwargs = {'upload_older_file': True, 'max_retries': max_retries, 'mode': mode}
    if do_parallel:
        processes = []
        for src_f, dbx_f in zip(src_files, dbx_files):
            p = mp.Process(target=upload_file, args=(dbx, src_f, dbx_f), kwargs=kwargs)
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    else:
        for src_f, dbx_f in zip(src_files, dbx_files):
            upload_file(dbx, src_f, dbx_f, **kwargs)


def create_directory_dbx(dbx, d, max_retries=10):
    """Creates a directory in Dropbox.
    """
    try:
        dbx.files_create_folder(d)
    except DropboxException as err:
        if err.error.is_path() and err.error.get_path().is_conflict() and err.error.get_path().get_conflict().is_folder():
            return
        else:
            retry(dbx.files_create_folder, d, max_retries=max_retries)


def create_directories_dbx(dbx, dirs, do_parallel=True, max_retries=10):
    """Create a list of directories in Dropbox.

    If do_parallel is True, the job is executed in parallel which reduces latency.
    """
    # TODO: Only create those that are not present already to decrease latency
    if do_parallel:
        processes = []
        for d in dirs:
            p = mp.Process(target=create_directory_dbx, args=(dbx, d), kwargs={'max_retries': max_retries})
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    else:
        for d in dirs:
            create_directory_dbx(dbx, d, max_retries=max_retries)


# def _handle_InternalServerError(err, max_retries=10, do_raise=False, dbx=None, f=None, path=None, mode=None, message=None):
#     if message is not None: message += "Internal server error."
#     if max_retries > 0 and all(v is not None for v in [dbx, f, path, mode]):
#         done = False
#         i = 0
#         while not done and i < max_retries:
#             time.sleep(1)
#             try:
#                 dbx.files_upload(f.read(), path, mode=mode)
#                 done = True
#             except DropboxException:
#                 pass
#             i += 1
#         if message is not None:
#             if done:
#                 message += " Succeded after " + str(i) + " retries."
#             else:
#                 message += " Failed after " + str(i) + " retries."
#     if message is not None: print(message)
#     if do_raise:
#         raise err
#     else:
#         return err


# def _handle_ApiError(err, do_raise=False, message=None):
#     """Handles Api errors.

#     Handles
#     - Insuffient space errors
#     - Write conflict errors
#     """
#     return err

#     if err.error.is_path() and err.error.get_path().is_insufficient_space():
#         if message is not None: message += 'Insufficient space.'
#     elif err.error.is_path() and err.error.get_path().is_conflict():
#         if err.error.get_path().get_conflict().is_folder():
#             # Folder already exists
#             pass
#         if err.error.get_path().get_conflict().is_file():
#             # File already exists and mode is not overwrite
#             if message is not None: message += "Write conflict on file."
#     elif err.error.is_path():
#         IPython.embed()
#     elif err.user_message_text:
#         if message is not None: message += err.user_message_text
#     else:
#         if message is not None: message += err
#     if message is not None: print(message)
#     if do_raise:
#         raise err
#     else:
#         return err
