import os
import argparse
from pathlib import Path
from typing import List
from vectorstore_manager import VectorstoreManager

def get_files_from_directory(directory: str, extensions: List[str] = None) -> List[Path]:
    """Get all files with specified extensions from a directory"""
    if extensions is None:
        extensions = ['.pdf', '.txt', '.md']
    
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    files = []
    for ext in extensions:
        files.extend(directory.glob(f"**/*{ext}"))
    
    return files

def create_vectorstore(
    store_path: str,
    input_paths: List[str],
    is_directory: bool = False
) -> None:
    """Create or update a vectorstore with files"""
    # Initialize manager
    manager = VectorstoreManager(store_path)
    
    # Process input paths
    files_to_process = []
    
    if is_directory:
        # If input is a directory, get all files
        for path in input_paths:
            files_to_process.extend(get_files_from_directory(path))
    else:
        # If input is individual files
        files_to_process = [Path(path) for path in input_paths]
    
    # Process files
    total_files = len(files_to_process)
    print(f"\nFound {total_files} files to process")
    
    for i, file_path in enumerate(files_to_process, 1):
        try:
            print(f"\nProcessing file {i}/{total_files}: {file_path}")
            manager.add_file(str(file_path))
            print(f"✅ Successfully added {file_path}")
        except Exception as e:
            print(f"❌ Error processing {file_path}: {str(e)}")
    
    # Save the final vectorstore
    manager.save()
    print(f"\nVectorstore created/updated at: {store_path}")

def main():
    parser = argparse.ArgumentParser(description="Create or update a FAISS vectorstore with documents")
    
    parser.add_argument(
        "input_paths",
        nargs="+",
        help="Paths to files or directories to process"
    )
    
    parser.add_argument(
        "-d", "--directory",
        action="store_true",
        help="Treat input paths as directories and process all supported files within them"
    )
    
    parser.add_argument(
        "-s", "--store-path",
        default="vector_storage",
        help="Path where the vectorstore will be saved (default: vector_storage)"
    )
    
    args = parser.parse_args()
    
    try:
        create_vectorstore(
            args.store_path,
            args.input_paths,
            args.directory
        )
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main() 