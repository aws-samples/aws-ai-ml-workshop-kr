from IPython.core.magic import register_cell_magic

#
# You can declare this in your Jupiter notebook. Mark cells like:
# %%write_and_run some.py 
# or
# %%write_and_run -a some.py
#
@register_cell_magic
def write_and_run(line, cell):
    argz = line.split()
    file = argz[-1]
    mode = 'w'
    if len(argz) == 2 and argz[0] == '-a':
        mode = 'a'
    with open(file, mode) as f:
        f.write(cell)
    get_ipython().run_cell(cell)
