import sys

import argparse
import importlib

commands = {
    'train': {
        'script': 'ocr4all_pixel_classifier.scripts.train',
        'main': 'main',
        'help': 'Train the neural network. See more via "* train --help"'
    },
    'predict': {
        'script': 'ocr4all_pixel_classifier.scripts.predict',
        'main': 'main',
        'help': 'Predict a result with the neural network. See more via "* predict --help"'
    },
    'predict-json': {
        'script': 'ocr4all_pixel_classifier.scripts.predict_json',
        'main': 'main',
        'help': 'Predict a result with the neural network, input via JSON. See more via "* predict --help"'
    },
    'create-dataset-file': {
        'script': 'ocr4all_pixel_classifier.scripts.create_dataset_file',
        'main': 'main',
        'help': 'Create a dataset file'
    },
    'compute-image-normalizations': {
        'script': 'ocr4all_pixel_classifier.scripts.compute_image_normalizations',
        'main': 'main',
        'help': 'Compute image normalizations'
    },
    'compute-image-map': {
        'script': 'ocr4all_pixel_classifier.scripts.generate_image_map',
        'main': 'main',
        'help': 'Generates color map'
    },
    'migrate-model': {
        'script': 'ocr4all_pixel_classifier.scripts.migrate_model',
        'main': 'main',
        'help': 'Convert old model to new format'
    },
    'gen-masks': {
        'script': 'ocr4all_pixel_classifier.scripts.gen_masks',
        'main': 'main',
        'help': 'Generate mask images from PAGE XML'
    },
}


def main():
    # Pretty print help for main programm
    usage = 'page-segmentation <command> [<args>]\n\nCOMMANDS:'
    # Add all commands to help
    max_name_length = max(len(name) for name, _ in commands.items())
    for name, command in commands.items():
        usage += '\n\t{name:<{col_width}}\t{help}'.format(name=name, col_width=max_name_length, help=command["help"])

    parser = argparse.ArgumentParser(usage=usage)
    parser.add_argument('command', help='The sub command to execute, see COMMANDS')

    args = parser.parse_args(sys.argv[1:2])
    sys.argv = sys.argv[:1] + sys.argv[2:]

    if args.command in commands.keys():
        command = commands[args.command]
        command_module = importlib.import_module(command['script'])
        command_main = getattr(command_module, command['main'])
        command_main()
    else:
        print('Unrecognized command')
        parser.print_help()
        exit(1)


if __name__ == "__main__":
    main()
