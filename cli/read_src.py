def get_source():
    source = open('./cli/starter_source.cuh', 'r').read()
    main_source = open('./src/main.cu', 'r').read()
    return source, main_source
