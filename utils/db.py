import os
import random
import time

import dropbox
import IPython
from dropbox.exceptions import ApiError, InternalServerError, DropboxException
from dropbox.files import WriteMode


def get_dropbox_client(token_file):
    with open(token_file) as f:
        token = f.readline()[:-1]
    return dropbox.Dropbox(token)


def upload_file(dbx, source_file, target_file, max_retries=10, mode=WriteMode('overwrite')):
    assert os.path.exists(source_file)
    with open(source_file, 'rb') as f:
        try:
            dbx.files_upload(f.read(), target_file, mode=mode)
        except ApiError as err:
            print("    Failed to upload " + source_file + ": ", end='')
            handle_ApiError(err)
        except InternalServerError as err:
            print("    Failed to upload " + source_file + ": ", end='')
            handle_InternalServerError(err, max_retries=max_retries, dbx=dbx, f=f, path=target_file, mode=mode)


def upload_directory(dbx, source_dir, target_dir, mode=WriteMode('overwrite')):
    """Copy an entire directory including all files and folders to Dropbox
    """
    print("Uploading directory to dropbox...")
    print("    Source: " + source_dir)
    print("    Target: " + target_dir)
    ns = 0
    for root, dirs, files in os.walk(source_dir):
        # Create all directories in this directory
        dirs = [os.path.join(target_dir, d) for d in dirs]
        create_directories(dbx, dirs)
        
        # Upload all files in this directory
        for filename in files:
            # Full local path
            local_path = os.path.join(root, filename)
            # Full dropbox path
            relative_path = os.path.relpath(local_path, source_dir)
            dropbox_path = os.path.join(target_dir, relative_path)
            # Put file
            print(" " * ns, end='\r')
            print("    Uploading: " + relative_path, end='\r')
            upload_file(dbx, local_path, dropbox_path, mode=mode)
            ns = 15 + len(relative_path)

    print(" " * ns, end='\r')
    print("    Uploading done")


def create_directories(dbx, dirs):
    """Create a list of directories in Dropbox
    """
    for d in dirs:
        try:
            dbx.files_create_folder(d)
        except ApiError as err:
            handle_ApiError(err)


def handle_InternalServerError(err, max_retries=10, do_raise=False, dbx=None, f=None, path=None, mode=None):
    print("Internal server error.")
    if max_retries > 0 and all(v is not None for v in [dbx, f, path, mode]):
        print("Trying again " + str(max_retries) + " times.")
        done = False
        i = 0
        while not done and i < max_retries:
            time.sleep(1)
            try:
                dbx.files_upload(f.read(), path, mode=mode)
                done = True
            except DropboxException:
                pass
            i += 1
    if do_raise:
        raise err
    else:
        return err


def handle_ApiError(err, do_raise=False):
    """Handles Api errors.

    Handles
    - Insuffient space errors
    - Write conflict errors
    """
    if err.error.is_path() and err.error.get_path().is_insufficient_space():
        print("Insufficient space.")
    elif err.error.is_path() and err.error.get_path().is_conflict():
        if err.error.get_path().get_conflict().is_folder():
            # Folder already exists
            # print("Folder already exists.")
        if err.error.get_path().get_conflict().is_file():
            # File already exists and mode is not overwrite
            print("Write conflict on file.")
    elif err.user_message_text:
        print(err.user_message_text)
    else:
        print(err)
    if do_raise:
        raise err
    else:
        return err
