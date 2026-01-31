import argparse
import os
import sys


FILE_SEP_TOKEN = '\n[[SEP_TOKEN]]\n'


def gather_contents(input_dir: str, output_file: str) -> None:
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for root, dirs, files in os.walk(input_dir):
            for filename in files:
                file_path = os.path.join(root, filename)
                rel_path = os.path.relpath(file_path, input_dir)

                try:
                    if filename.lower().endswith('.md'): # only gathering context from md files
                        out_f.write(f"{FILE_SEP_TOKEN}{rel_path}:\n")
                        with open(file_path, 'r', encoding='utf-8') as in_f:
                            out_f.write(in_f.read() + '\n\n')
                except UnicodeDecodeError:
                    raise Exception(f'Failed to decode file {filename} as utf-8')
                except Exception as e:
                    raise Exception(f'Failed to read file {filename}:\n{e}')


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gathers the contents of directory files: text files, and .class files via javap"
    )
    parser.add_argument('input_dir', help='Path to the directory')
    parser.add_argument('output_file', help='Path to the output file')
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Error: directory '{args.input_dir}' not found.", file=sys.stderr)
        sys.exit(1)

    gather_contents(args.input_dir, args.output_file)
    print(f"File contents from '{args.input_dir}' have been written to '{args.output_file}'")


def create_txt_db():
    gather_contents('./Main vault', 'all_data.txt')


if __name__ == '__main__':
    #main()
    create_txt_db()
