# folder_mount.py

import sys
import gdown


def mount_file(file_id, output="driveFile"):
    """
    Download a file from Google Drive using its file ID.
    """

    print(f"Downloading Google Drive file: {file_id}")

    try:
        downloaded = gdown.download(
            id=file_id,
            output=output,
            quiet=False
        )

        if downloaded:
            print(f"File downloaded successfully: {downloaded}")
        else:
            print("Download failed.")

    except Exception as e:
        print(f"Error while downloading file: {e}")


def mount_folder(folder_id, output="driveDownloaded_folder"):
    """
    Download a Google Drive folder recursively.
    """

    url = f"https://drive.google.com/drive/folders/{folder_id}"

    print(f"Downloading Google Drive folder: {folder_id}")

    try:
        downloaded = gdown.download_folder(
            url,
            output=output,
            quiet=False
        )

        print(f"Folder downloaded successfully: {downloaded}")

    except Exception as e:
        print(f"Error while downloading folder: {e}")


def resolve_file():
    """
    Usage:
        python folder_mount.py <file_id>
        python folder_mount.py <file_id> <output>
    """

    if len(sys.argv) == 3:
        mount_file(sys.argv[1], sys.argv[2])

    elif len(sys.argv) == 2:
        mount_file(sys.argv[1])

    else:
        print("Usage:")
        print("  python folder_mount.py <file_id>")
        print("  python folder_mount.py <file_id> <output>")


def resolve_folder():
    """
    Usage:
        python folder_mount.py <folder_id>
        python folder_mount.py <folder_id> <output>
    """

    if len(sys.argv) == 3:
        mount_folder(sys.argv[1], sys.argv[2])

    elif len(sys.argv) == 2:
        mount_folder(sys.argv[1])

    else:
        print("Usage:")
        print("  python folder_mount.py <folder_id>")
        print("  python folder_mount.py <folder_id> <output>")


if __name__ == "__main__":

    # Change this depending on what you want to mount.

    # For FILE:
    # resolve_file()

    # For FOLDER:
    resolve_folder()