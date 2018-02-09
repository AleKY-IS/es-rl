import os
import time

import dropbox
from dropbox.exceptions import ApiError, InternalServerError
from dropbox.files import WriteMode

import random


def get_dropbox_client(token_file):
    with open(token_file) as f:
        token = f.readline()[:-1]
    return dropbox.Dropbox(token)


def copy_dir(dbx, source_dir, target_dir, mode='overwrite', silent=False):
    print("Uploading directory to dropbox...")
    print("    Source: " + source_dir)
    print("    Target: " + target_dir)
    for root, dirs, files in os.walk(source_dir):
        # Create directories (includes empty ones)
        for d in dirs:
            dbx.file_create_folder(dirs)
        # Put all files
        for filename in files:
            # Full local path
            local_path = os.path.join(root, filename)
            # Full dropbox path
            relative_path = os.path.relpath(local_path, source_dir)
            dropbox_path = os.path.join(target_dir, relative_path)
            # Put file
            with open(local_path, 'rb') as f:
                try:
                    print("    Uploading: " + relative_path, end='\r')
                    dbx.files_upload(f.read(), dropbox_path, mode=WriteMode('overwrite'))
                except ApiError as err:
                    # This checks for the specific error where a user doesn't have
                    # enough Dropbox space quota to upload this file
                    if err.error.is_path() and err.error.get_path().reason.is_insufficient_space():
                        print("    Failed to copy " + relative_path + ". Insufficient space.")
                    if err.error.is_path() and err.error.get_path().reason.is_conflict():
                        print("    Failed to copy " + relative_path + ". Write conflict.")
                    elif err.user_message_text:
                        print(err.user_message_text)
                    else:
                        print(err)
                except InternalServerError as err:
                    print("    Failed to upload " + relative_path + ". Internal server error. Trying again. ", end='')
                    time.sleep(1)
                    try:
                        dbx.files_upload(f.read(), dropbox_path, mode=WriteMode('overwrite'))
                        print("Success.")
                    except:
                        print("Failed.")
