import re
import click
from cli import get_source


def camel_case(string: str):
    temp = string.split('_')
    return ''.join([t.capitalize() for t in temp])


@click.command()
@click.option('--kernel_name', '-k')
def main(kernel_name):
	header_source, main_source = get_source()
	class_name = camel_case(kernel_name)
	source_code = header_source.replace('hello_world_kernel', kernel_name + '_kernel')
	source_code = source_code.replace('HelloWorld', class_name)
	search = re.findall(r'include\s\"headers/([\w\d]+)\.cuh\"', string=main_source)[0]
	main_source = main_source.replace(search, class_name)
	with open('./src/headers/{}.cuh'.format(class_name), 'w') as out_file:
		out_file.write(source_code)
	with open('./src/main.cu', 'w') as out_file:
		out_file.write(main_source)


if __name__ == "__main__":
    main()
