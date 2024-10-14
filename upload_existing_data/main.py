from dotenv import load_dotenv
import csv
import paramiko
import zipfile
from pymongo import MongoClient
import re
import uuid
from bson import ObjectId

import pandas as pd
import os

load_dotenv()

def parse_board_corners(corner_str):
    corner_str = corner_str.strip()[1:-1]
    lines = corner_str.split('\n')
    corners = []
    for line in lines:
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", line)
        if len(nums) == 2:
            corners.append([float(nums[0]), float(nums[1])])
    return corners


def create_data_in_db(csv_file_path, remote_folder, match_id):
    df = pd.read_csv(csv_file_path)

    #filtered_df = df[df['file'].str.contains(match_id)]
    #filtered_df = df[df['file'].str.match(fr'.*{match_id}$')]
    filtered_df = df[df['file'].str.contains(r'WC-Round18\\WC-Round18\b', regex=True)]

    total_rows = filtered_df.shape[0]
    print("Total rows:", total_rows)


    moves = []
    non_existent_files = 0  # Counter for files that do not exist

    for _, row in filtered_df.iterrows():
        image_name = row['file'].split("\\")[-1]
        file = f"{remote_folder}/{match_id}/{image_name}"
        file_path = os.path.join(os.getenv("LOCAL_FOLDER_PATH"), image_name)

        if os.path.exists(file_path):
            board_corners_str = None if pd.isna(row['board_corners']) else row['board_corners']

            move = {
                "_id": ObjectId(),
                "file": file,
                "fen": row['fen'],
                "orientation": row['orientation'],
                "board_corners": board_corners_str,
                "verified": False,
                "source": row['source'],
            }
            moves.append(move)
        else:
            print("File does not exist:", file_path)
            non_existent_files += 1

    total_moves = len(moves)
    print("Total moves being inserted:", total_moves)
    print("Total nonexistent files:", non_existent_files)

    match = matches_collection.find_one({'match_id': match_id})
    if not match:
        new_match = {
            'match_id': match_id,
            'source': moves[0]['source'] if moves else 'test',
            'total_moves': total_moves,
            'moves': moves
        }
        matches_collection.insert_one(new_match)
    else:
        matches_collection.update_one(
            {'match_id': match_id},
            {'$set': {'moves': moves}}
        )


def zip_images(directory_path, zip_file_path):
    png_files = [f for f in os.listdir(directory_path) if f.endswith('.png')]
    if png_files:  # Check if there are any .png files
        with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for filename in png_files:
                zipf.write(os.path.join(directory_path, filename), arcname=filename)
        print("ZIP file created successfully with the .png files.")
    else:
        print("No .png files found in the directory. ZIP file was not created.")


def upload_and_unzip_file_on_vm(local_path, remote_match_folder, remote_zip_path, ssh_host, ssh_user, ssh_password):
    try:
        with paramiko.SSHClient() as ssh:
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(ssh_host, username=ssh_user, password=ssh_password)

            # Create the match-specific directory on the VM
            stdin, stdout, stderr = ssh.exec_command(f"mkdir -p {remote_match_folder}")
            stdout.channel.recv_exit_status()  # Wait for command to complete

            with ssh.open_sftp() as sftp:
                sftp.put(local_path, remote_zip_path)
                print(f"Uploaded {local_path} to {remote_zip_path}")

            # Unzip the file on the remote server
            stdin, stdout, stderr = ssh.exec_command(f"unzip -o {remote_zip_path} -d {remote_match_folder}")
            stdout.channel.recv_exit_status()  # Wait for command to complete
            print("Unzip successful")

            # Delete the ZIP file after successful extraction
            ssh.exec_command(f"rm {remote_zip_path}")
            print("ZIP file deleted successfully")

    except Exception as e:
        print(f"An error occurred while uploading the images to the vm: {e}")


if __name__ == "__main__":
    directory_path = os.getenv("LOCAL_FOLDER_PATH")
    remote_folder = os.getenv("VM_USER_PATH")+os.getenv("VM_FOLDER")

    remote_folder_without_user = os.getenv("VM_FOLDER")

    zip_file_path = os.path.join(directory_path, 'images.zip')

    # MongoDB connection details
    mongo_connection_string = os.getenv("MONGO_DB_CONNECTION_STRING")
    mongo_db= os.getenv("MONGO_DB")
    mongo_collection= os.getenv("MONGO_COLLECTION")
    mongo_client = MongoClient(mongo_connection_string)
    db = mongo_client[mongo_db]
    matches_collection = db[mongo_collection]

    # SSH details
    ssh_host = os.getenv("VM_HOST")
    ssh_user = os.getenv("VM_USERNAME")
    ssh_password = os.getenv("VM_PASSWORD")

    #match_id = os.path.basename(directory_path)
    #print("match_id", match_id)
    #match_id = match_id.split("match-")[1]


    match_id = "WC-Round18"

    csv_file_path = os.getenv("CSV_FILE_PATH")

    #remote_match_folder = f"{remote_folder}/{match_id}"
    remote_match_folder = f"{remote_folder}/{match_id}"
    #remote_zip_path = f"{remote_match_folder}/{match_id}.zip"
    remote_zip_path = f"{remote_folder}/{match_id}.zip"

    print("====================================================================")
    print("Uploading images to the vm ...")
    #zip_images(directory_path, zip_file_path)
    upload_and_unzip_file_on_vm(os.getenv("ZIP_FILE_PATH"), remote_folder, remote_zip_path, ssh_host, ssh_user, ssh_password)
    print('Upload complete.')

    print("====================================================================")
    print("Adding data to the DB...")
    create_data_in_db(csv_file_path, f"{remote_folder_without_user}", match_id)
    print("Process finished")
