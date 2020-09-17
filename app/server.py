import aiohttp
import asyncio
import uvicorn
from fastai import *
from fastai.vision import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
import requests
import os

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    
    print('Complete')
    
def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for index, chunk in enumerate(response.iter_content(CHUNK_SIZE)):
            print(index)
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
    print('Complete')

#export_file_url = 'https://drive.google.com/file/d/1tFxujqMyii2hCEoaIOUJ-QJ2eryYimI2/view?usp=sharing'
export_file_url = 'https://www.dropbox.com/s/zim9d5cbrd8cxcu/instagram.pkl?dl=0'
export_file_name = 'instagram.pkl'



classes = ['NO', 'YES']
path = Path(__file__).parent

if os.path.exists(os.path.join(path,export_file_name)):
    pass
else:
    download_file_from_google_drive('1EhOJZM3hVmphfCGDQU3Ik1mjVKYSbllY', path / export_file_name)

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    #await download_file(export_file_url, path / export_file_name)
    #download_file_from_google_drive('1EhOJZM3hVmphfCGDQU3Ik1mjVKYSbllY', path / export_file_name)

    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    img = open_image(BytesIO(img_bytes))
    img = img.resize(torch.Size([img.shape[0],224, 224]))#.data.half()
    #img = Image(img)
    prediction = learn.to_fp32().predict(img)[0]
    print(str(prediction))
    
    r = str(prediction)
    if r == '1':
        r = 'Cool Pic'
    else:
        r = 'Not Cool'
    return JSONResponse({'result': r})


if __name__ == '__main__':
    #if 'serve' in sys.argv:
    uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
