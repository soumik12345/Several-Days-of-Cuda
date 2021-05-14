import os
import click


@click.command()
@click.option('--kernel_name', '-k', help='Kernel Name')
@click.option('--source_name', '-s', help='Source Name')
@click.option('--memory_allocation_auto', '-a', default=False, help='Enable Automatic Memory Allocation')
def main(kernel_name, source_name, memory_allocation_auto):
    template_path = './src/lib/BasicExamples/'
    template_source = open(template_path + 'ArrayManipulation.cuh').read() \
        if memory_allocation_auto else open(template_path + 'ArrayManipulationManualMemoryAllocation.cuh').read()
    source_code = template_source.replace('array_manipulation_kernel', kernel_name + '_kernel')
    source_code = source_code.replace(
        'ArrayManipulation', source_name.split('/')[-1] if '/' in source_name else source_name)
    main_source = open('./src/main.cu').read().split('\n')
    main_source[0] = '#include "lib/{}.cuh"'.format(source_name)
    main_source = '\n'.join(main_source)
    if '/' in source_name:
        os.system('mkdir -p {}'.format('./src/lib/' + source_name.split('/')[0]))
    with open('./src/lib/{}.cuh'.format(source_name), 'w') as out_file:
        out_file.write(source_code)
    print('New template ready at {}'.format('./src/lib/{}.cuh'.format(source_name)))
    with open('./src/main.cu', 'w') as out_file:
        out_file.write(main_source)


if __name__ == "__main__":
    main()
