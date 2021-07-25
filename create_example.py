import os
import click


@click.command()
@click.option('--example_name', '-k', help='Example Name in Camel Case')
def main(example_name):
    cuda_template_path = './src/array-manipulation/main.cu'
    cmake_template_path = './src/array-manipulation/CMakeLists.txt'
    outer_cmake_path = './CMakeLists.txt'
    example_name_snake_case = ''.join(['_' + i.lower() if i.isupper() else i for i in example_name]).lstrip('_')
    try:
        os.mkdir(os.path.join('./src', example_name_snake_case.replace('_', '-')))
    except Exception:
        pass
    cuda_template_source = open(cuda_template_path).read().replace('ArrayManipulation', example_name)
    cuda_template_source = cuda_template_source.replace('array_manipulation_kernel', example_name_snake_case)
    cmake_template_source = open(cmake_template_path).read().replace('array_manipulation', example_name_snake_case)
    outer_cmake_source = open(outer_cmake_path).read()
    outer_cmake_source += '\nadd_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/' \
                          + example_name_snake_case.replace('_', '-') + ')\n'
    outer_cmake_source += 'add_custom_target({})\n'.format(example_name_snake_case.replace('_', '-'))
    outer_cmake_source += 'add_dependencies({} {})\n'.format(
        example_name_snake_case.replace('_', '-'), example_name_snake_case)
    with open(os.path.join('./src', example_name_snake_case.replace('_', '-'), 'main.cu'), 'w') as f:
        f.write(cuda_template_source)
    with open(os.path.join('./src', example_name_snake_case.replace('_', '-'), 'CMakeLists.txt'), 'w') as f:
        f.write(cmake_template_source)
    with open(outer_cmake_path, 'w') as f:
        f.write(outer_cmake_source)


if __name__ == "__main__":
    main()
