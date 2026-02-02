import argparse
import sys
from separator import SpeakerSeparator


def main():
    parser = argparse.ArgumentParser(
        description='Separate two speakers from an audio file into separate WAV files'
        )
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to input WAV file containing two speakers'
    )
    # optional argument for output directory
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./output',
        help='Output directory for separated files (default: ./output)'
    )
    
    args = parser.parse_args()
    
    try:
        #initialize and perform seperation
        separator = SpeakerSeparator()
        separator.separate(args.input_file, args.output_dir)
        print("Speaker separation completed.")

    # handle errors 
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n  Interrupted by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f" Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()