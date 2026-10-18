"""Hash and validate a complete source/data/result archive without environment caches."""
from pathlib import Path
import hashlib,json
from zipfile import ZipFile,ZIP_DEFLATED
ROOT=Path(__file__).resolve().parents[1]
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
    return h.hexdigest()
def include(p):
    rel=p.relative_to(ROOT)
    return p.is_file() and not any(x in {'.git','.venv','__pycache__','.ipynb_checkpoints','pdf_qa'} for x in rel.parts) and p.suffix.lower() not in ('.zip','.partial') and p.name not in ('SHA256SUMS.txt','archive_validation.json','packaging.log')
def run():
    files=sorted(p for p in ROOT.rglob('*') if include(p))
    manifest=ROOT/'SHA256SUMS.txt'; hashes={p.relative_to(ROOT).as_posix():sha(p) for p in files}
    manifest.write_text(''.join(f'{h}  {name}\n' for name,h in hashes.items()),encoding='utf-8')
    archive=ROOT/'Research_Project_Prime.zip'
    with ZipFile(archive,'w',compression=ZIP_DEFLATED,compresslevel=6) as z:
        for p in files+[manifest]: z.write(p,arcname='Internship_Research_Prime/'+p.relative_to(ROOT).as_posix())
    with ZipFile(archive) as z:
        assert z.testzip() is None
        for name,h in hashes.items(): assert hashlib.sha256(z.read('Internship_Research_Prime/'+name)).hexdigest()==h
        assert z.read('Internship_Research_Prime/SHA256SUMS.txt')==manifest.read_bytes()
    result={'archived_files':len(files)+1,'archive_bytes':archive.stat().st_size,'archive_sha256':sha(archive),
            'all_entry_hashes_verified':True,'zip_crc_verified':True,'recursive_archive_excluded':True}
    (ROOT/'results/archive_validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
if __name__=='__main__':run()
