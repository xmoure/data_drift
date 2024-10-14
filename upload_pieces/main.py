from dotenv import load_dotenv
import csv
import paramiko
import zipfile
from pymongo import MongoClient
import re
import uuid
from bson import ObjectId
import bson

import pandas as pd
import os
from PIL import Image
from constants import Task, ChessPiece, ChessAbbreviation, Color

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


def count_files(directory):
    return len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])


def process_images(local_path, color, piece_type, model_type, split, vm_folder, chess_abbreviation):
    inserted_count = 0
    updated_count = 0
    # Loop through each file in the directory
    total = count_files(local_path)
    print("Number of files:", total)
    number_files = 0

    count_find = 0
    count_not_found = 0

    for filename in os.listdir(local_path):
        number_files += 1
        print(f"{number_files} out of {total} ")
        # Check if the file is an image based on its extension
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            print("filename", filename)

            base_name = '_'.join(filename.split('_')[:-1])
            position = filename.split('_')[-1][:-4]
            # Print or store the extracted base name
            #print("base_name",base_name)
            parts = base_name.split("_", 1)
            #print("parts_0", parts[0])
            source = parts[0]
            print("source", source)
            print("parts1", parts[1])
            #print("parts_1",parts[1])
            #print("position", position)
            # Properly escape any regex special characters in the base_filename
            base_filename_escaped = re.escape(parts[1])
            #print("base name escaped", base_filename_escaped)
            # Construct full file path
            file_path = os.path.join(local_path, filename)
            regex = re.compile(f".*{re.escape(parts[1])}.*")



            print("regex", regex)

            pipeline = [
                {
                    "$match": {
                        "source": source,
                        "moves.file": {
                            "$regex": regex
                        }
                    }
                },
                {
                    "$project": {
                        "moves": {
                            "$filter": {
                                "input": "$moves",
                                "as": "move",
                                "cond": {"$regexMatch": {
                                    "input": "$$move.file",
                                    "regex": regex
                                }}
                            }
                        },
                        "match_id": 1,
                        "_id": 1
                    }
                }
            ]


            results = matches_collection.aggregate(pipeline)
            # Initialize a counter
            count = 0


            # Iterate through results and process
            for result in results:
                #print("result", result['_id'])
                match_id = result['_id']
                move_id = result['moves'][0]['_id']
                #print("move_id", move_id)
                print("file", f"{vm_folder}{model_type}/{split}/{chess_abbreviation}/{filename}")
                count += 1
                new_piece = {
                    'file': f"{vm_folder}{model_type}/{split}/{chess_abbreviation}/{filename}",
                    'color': color,
                    'piece_type': piece_type,
                    'position': position,
                }

                #piece = pieces_collection.find_one({'match_id': match_id, 'move_id': move_id})

                field_to_update = "occupancy" if model_type == Task.OCCUPANCY_CLASSIFIER else "piece"

                f = f"{vm_folder}{model_type}/{split}/{chess_abbreviation}/{filename}"

                occupancy_piece = pieces_collection.find_one(
                    {"occupancy.file": f},
                    {"piece.$": 1}
                )

                if occupancy_piece:
                    print("Query returned results:", occupancy_piece)
                    count_find += 1
                    continue
                else:
                    print("No results found.")
                    count_not_found += 1




                # Update or insert new piece
                update_result = pieces_collection.update_one(
                    {'match_id': match_id, 'move_id': move_id},
                    {"$push": {field_to_update: new_piece}},
                    upsert=True  # This creates a new document if no existing document matches
                )

                if update_result.upserted_id:
                    inserted_count += 1
                else:
                    updated_count += update_result.modified_count

    print("count_find", count_find)
    print("count_not_found", count_not_found)
    print("inserted_count", inserted_count)
    print("updated_count", updated_count)
    print("-----------------------------------------------------------------------------------------------")

    # Add any additional image processing here




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

            print("Checking if file exists:", os.path.exists(local_path))
            print("Is it a file:", os.path.isfile(local_path))

            # Create the match-specific directory on the VM
            stdin, stdout, stderr = ssh.exec_command(f"mkdir -p {remote_match_folder}")
            stdout.channel.recv_exit_status()  # Wait for command to complete

            print("local_path", local_path)
            print("remote_match_folder", remote_match_folder)
            print("remote zip file", remote_zip_path)

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



def upload_and_unzip_file_on_vm_v2(local_path, remote_match_folder, remote_zip_path, ssh_host, ssh_user, ssh_password):
    try:
        with paramiko.SSHClient() as ssh:
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            # Set a timeout for the connection attempt
            ssh.connect(ssh_host, username=ssh_user, password=ssh_password, timeout=120)  # 30 seconds timeout

            print("Checking if file exists:", os.path.exists(local_path))
            print("Is it a file:", os.path.isfile(local_path))

            # Create the match-specific directory on the VM
            stdin, stdout, stderr = ssh.exec_command(f"mkdir -p {remote_match_folder}")
            if stdout.channel.recv_exit_status() != 0:
                raise Exception("Failed to create directory on VM: " + stderr.read().decode())

            print("local_path:", local_path)
            print("remote_match_folder:", remote_match_folder)
            print("remote zip file:", remote_zip_path)

            with ssh.open_sftp() as sftp:
                sftp.put(local_path, remote_zip_path)
                print(f"Uploaded {local_path} to {remote_zip_path}")

            # Unzip the file on the remote server
            stdin, stdout, stderr = ssh.exec_command(f"unzip -o {remote_zip_path} -d {remote_match_folder}")
            if stdout.channel.recv_exit_status() != 0:
                raise Exception("Unzip failed: " + stderr.read().decode())
            print("Unzip successful")

            # Delete the ZIP file after successful extraction
            stdin, stdout, stderr = ssh.exec_command(f"rm {remote_zip_path}")
            if stdout.channel.recv_exit_status() != 0:
                raise Exception("Failed to delete ZIP file: " + stderr.read().decode())
            print("ZIP file deleted successfully")

    except paramiko.ssh_exception.NoValidConnectionsError as nvce:
        print(f"Connection error: {nvce}")
    except paramiko.AuthenticationException as ae:
        print(f"Authentication failed: {ae}")
    except paramiko.SSHException as se:
        print(f"SSH issue: {se}")
    except Exception as e:
        print(f"An error occurred: {e}")



""" def test(local_path, color, piece_type, model_type, split, vm_folder, chess_abbreviation):
    filename = "rendered_3777_e8.png"
    # Loop through each file in the directory
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        print("filename", filename)

        base_name = '_'.join(filename.split('_')[:-1])
        position = filename.split('_')[-1][:-4]
        # Print or store the extracted base name
        print("base_name",base_name)
        parts = base_name.split("_", 1)
        print("parts_0", parts[0])
        source = parts[0]
        print("parts_1",parts[1])
        print("position", position)
        # Properly escape any regex special characters in the base_filename
        base_filename_escaped = re.escape(parts[1])
        print("base name escaped", base_filename_escaped)
        # Construct full file path
        file_path = os.path.join(local_path, filename)
        regex = re.compile(f".*{re.escape(parts[1])}.*")


        pipeline = [
                {
                    "$match": {
                        "source": source,
                        "moves.file": {
                            "$regex": regex
                        }
                    }
                },
                {
                    "$project": {
                        "moves": {
                            "$filter": {
                                "input": "$moves",
                                "as": "move",
                                "cond": {"$regexMatch": {
                                    "input": "$$move.file",
                                    "regex": regex
                                }}
                            }
                        },
                        "match_id": 1,
                        "_id": 1
                    }
                }
        ]


        results = matches_collection.aggregate(pipeline)
            # Initialize a counter
        count = 0

            # Iterate through results and process
        for result in results:
            count += 1
            print("result", result['_id'])
            match_id = result['_id']
            move_id = result['moves'][0]['_id']
            print("move_id", move_id)
            print("file", f"{vm_folder}{model_type}/{split}/{chess_abbreviation}/{filename}")
        print("count", count)

    print("-----------------------------------------------------------------------------------------------")

 """

if __name__ == "__main__":
    directory_path = os.getenv("LOCAL_FOLDER_PATH")
    remote_folder = os.getenv("VM_USER_PATH")+os.getenv("VM_FOLDER")

    remote_folder_without_user = os.getenv("VM_FOLDER")

    zip_file_path = os.path.join(directory_path, 'images.zip')

    # MongoDB connection details
    mongo_connection_string = os.getenv("MONGO_DB_CONNECTION_STRING")
    mongo_db= os.getenv("MONGO_DB")
    mongo_collection_matches= os.getenv("MONGO_COLLECTION_MATCHES")
    mongo_collection_pieces= os.getenv("MONGO_COLLECTION_PIECES")
    mongo_client = MongoClient(mongo_connection_string)
    db = mongo_client[mongo_db]
    matches_collection = db[mongo_collection_matches]
    pieces_collection = db[mongo_collection_pieces]

    # SSH details
    ssh_host = os.getenv("VM_HOST")
    ssh_user = os.getenv("VM_USERNAME")
    ssh_password = os.getenv("VM_PASSWORD")

    #match_id = os.path.basename(directory_path)
    #print("match_id", match_id)
    #match_id = match_id.split("match-")[1]


    match_id = "processed"

    csv_file_path = os.getenv("CSV_FILE_PATH")

    #remote_match_folder = f"{remote_folder}/{match_id}"
    remote_match_folder = f"{remote_folder}/{match_id}"
    #remote_zip_path = f"{remote_match_folder}/{match_id}.zip"
    remote_zip_path = f"{remote_folder}/{match_id}.zip"


    process_images(directory_path, Color.EMPTY, ChessPiece.EMPTY, Task.OCCUPANCY_CLASSIFIER, "split1", remote_folder_without_user, ChessAbbreviation.EMPTY)
    # r = os.getenv("VM_USER_PATH")+os.getenv("VM_FOLDER")
    # print("====================================================================")
    # print("Uploading images to the vm ...")
    # #zip_images(directory_path, zip_file_path)
    # upload_and_unzip_file_on_vm_v2(os.getenv("ZIP_FILE_PATH"), remote_folder, remote_zip_path, ssh_host, ssh_user, ssh_password)
    # print('Upload complete.')

    """ print("====================================================================")
    print("Adding data to the DB...")
    create_data_in_db(csv_file_path, f"{remote_folder_without_user}", match_id)
    print("Process finished") """

