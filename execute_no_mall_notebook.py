"""Convert the percent-cell source to a standalone notebook and execute every cell."""
from pathlib import Path
import sys, os, io, base64, traceback, contextlib
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'.annual_packages'))
os.environ['MPLBACKEND']='Agg'
os.environ['OMP_NUM_THREADS']='4'
import nbformat

source=(ROOT/'annual_no_mall.py').read_text(encoding='utf-8')
cells=[]; buffer=[]; kind=None
def append_cell():
    if kind is None: return
    content='\n'.join(buffer).strip()
    if kind=='markdown':
        content='\n'.join(line[2:] if line.startswith('# ') else (line[1:] if line.startswith('#') else line) for line in content.splitlines())
        cells.append(nbformat.v4.new_markdown_cell(content))
    else: cells.append(nbformat.v4.new_code_cell(content))
for line in source.splitlines():
    if line.startswith('# %%'):
        append_cell(); kind='markdown' if '[markdown]' in line else 'code'; buffer=[]
    else: buffer.append(line)
append_cell()
notebook=nbformat.v4.new_notebook(cells=cells,metadata={
    'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
    'language_info':{'name':'python','version':sys.version.split()[0]},
    'application/vnd.databricks.v1+notebook':{'language':'python','notebookName':'sales-forecast-no-mall','notebookMetadata':{'pythonIndentUnit':4}},
})
path=ROOT/'sales-forecast-no-mall.ipynb'
nbformat.write(notebook,path)
namespace={'__name__':'__main__'}; outputs=[]; count=0
class Tee(io.StringIO):
    def write(self,text):
        sys.__stdout__.write(text); sys.__stdout__.flush()
        return super().write(text)
def display(*values):
    for value in values:
        data={'text/plain':str(value)}
        if hasattr(value,'_repr_html_'): data['text/html']=value._repr_html_()
        outputs.append(nbformat.v4.new_output('display_data',data=data,metadata={}))
namespace['display']=display
def show(*args,**kwargs):
    import matplotlib.pyplot as plt
    for number in plt.get_fignums():
        fig=plt.figure(number); stream=io.BytesIO(); fig.savefig(stream,format='png',dpi=120,bbox_inches='tight')
        outputs.append(nbformat.v4.new_output('display_data',data={'image/png':base64.b64encode(stream.getvalue()).decode('ascii'),'text/plain':'<Figure>'},metadata={}))
    plt.close('all')
for index,cell in enumerate(notebook.cells):
    if cell.cell_type!='code': continue
    count+=1; outputs=[]; captured=Tee()
    print(f'Executing cell {count} ({index+1}/{len(notebook.cells)})',flush=True)
    try:
        with contextlib.redirect_stdout(captured):
            exec(compile(cell.source,f'{path.name}:cell-{count}','exec'),namespace)
        if 'plt' in namespace: namespace['plt'].show=show
        if captured.getvalue(): outputs.insert(0,nbformat.v4.new_output('stream',name='stdout',text=captured.getvalue()))
        cell.execution_count=count; cell.outputs=outputs
    except Exception as error:
        cell.execution_count=count
        cell.outputs=outputs+[nbformat.v4.new_output('error',ename=type(error).__name__,evalue=str(error),traceback=traceback.format_exc().splitlines())]
        nbformat.write(notebook,path)
        raise
    nbformat.write(notebook,path)
nbformat.validate(notebook)
print(f'Executed and validated {count} code cells; saved {path}',flush=True)
