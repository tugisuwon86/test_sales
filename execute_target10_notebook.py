from pathlib import Path
import sys,io,contextlib,traceback,os
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'.annual_packages'))
os.environ['OMP_NUM_THREADS']='4'
import nbformat
from threadpoolctl import threadpool_limits
path=ROOT/'sales-forecast-target10-geography.ipynb'
notebook=nbformat.read(path,as_version=4)
namespace={'__name__':'__main__'}
count=0
with threadpool_limits(limits=4):
    for cell in notebook.cells:
        if cell.cell_type!='code': continue
        count+=1; stream=io.StringIO()
        print('Executing geographic notebook cell',count,flush=True)
        try:
            with contextlib.redirect_stdout(stream): exec(compile(cell.source,f'cell-{count}','exec'),namespace)
            cell.execution_count=count
            cell.outputs=[nbformat.v4.new_output('stream',name='stdout',text=stream.getvalue())] if stream.getvalue() else []
        except Exception as error:
            cell.execution_count=count
            cell.outputs=[nbformat.v4.new_output('error',ename=type(error).__name__,evalue=str(error),traceback=traceback.format_exc().splitlines())]
            nbformat.write(notebook,path); raise
        nbformat.write(notebook,path)
nbformat.validate(notebook)
print('Executed',count,'code cells successfully',flush=True)
