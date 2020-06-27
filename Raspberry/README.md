### A IMAGEM QUE VC ACABOU DE INSTALAR NA RASPBERRY JÁ ESTÁ CONFIGURADA COM:

- 2020-05-27-raspios-buster-lite-armhf.img (Debian 10)

- Configurações de local, wifi e tudo o mais ajustados para São Paulo

- SSH habilitado

- GPIO remoto habilitado

- Git já está instalado e o repositório da IC já foi clonado

- Conecte-se a uma rede wifi ou cabeada para acessar o SSH. Se for wifi é só entrar em "sudo raspi-config"

### EM RELAÇÃO AO RESPEAKER E AO DRIVER:

- O driver do ReSpeaker já foi instalado. Até o passo 5 desse link já está tudo pronto: https://wiki.seeedstudio.com/ReSpeaker_4_Mic_Array_for_Raspberry_Pi/#extract-voice

- O comando arecord está funcionando normalmente

### EM RELAÇÃO AO PYTHON, OS SEGUINTES PASSOS FORAM TOMADOS:

- Python 3.6.1 instalado com pyenv

- Há um ambiente virtual chamado venv1. Para ativar -> pyenv activate venv1

- Nesse ambiente virtual, temos os seguintes pacotes instalados (pip list):

attrs              19.3.0

audioread          2.1.8

backcall           0.2.0

bleach             3.1.5

cffi               1.14.0

cycler             0.10.0

Cython             0.29.20

decorator          4.4.2

defusedxml         0.6.0

entrypoints        0.3

importlib-metadata 1.7.0

ipykernel          5.3.0

ipython            7.16.1

ipython-genutils   0.2.0

ipywidgets         7.5.1

jedi               0.17.1

Jinja2             2.11.2

joblib             0.15.1

jsonschema         3.2.0

jupyter            1.0.0

jupyter-client     6.1.3

jupyter-console    6.1.0

jupyter-core       4.6.3

kiwisolver         1.2.0

librosa            0.7.2

llvmlite           0.32.0

MarkupSafe         1.1.1

matplotlib         3.2.2

mistune            0.8.4

nbconvert          5.6.1

nbformat           5.0.7

notebook           6.0.3

numba              0.49.0

numpy              1.19.0

packaging          20.4

pandas             1.0.5

pandocfilters      1.4.2

parso              0.7.0

pexpect            4.8.0

pickleshare        0.7.5

pip                20.1.1

prometheus-client  0.8.0

prompt-toolkit     3.0.5

ptyprocess         0.6.0

PyAudio            0.2.11

pycparser          2.20

Pygments           2.6.1

pyparsing          2.4.7

pyrsistent         0.16.0

python-dateutil    2.8.1

pytz               2020.1

pyzmq              19.0.1

qtconsole          4.7.5

QtPy               1.9.0

resampy            0.2.2

scikit-learn       0.23.1

scipy              1.5.0

Send2Trash         1.5.0

setuptools         28.8.0

six                1.15.0

SoundFile          0.10.3.post1

terminado          0.8.3

testpath           0.4.4

threadpoolctl      2.1.0

tornado            6.0.4

traitlets          4.3.3

wcwidth            0.2.5

webencodings       0.5.1

wheel              0.34.2

widgetsnbextension 3.5.1

zipp               3.1.0

### PASSOS PARA CONSEGUIR DEIXAR OS PACOTES FUNCIONANDO:

- O RASPBERRY PI OS COM DEBIAN 10 VEM COM O PYTHON 3.7. NÃO ROLOU USAR O PIP DIREITO. VAMOS INSTALAR O PYTHON 3.6 PRA INSTALAR O PYENV TEM QUE INSTALAR ISSO AQUI PRIMEIRO:
sudo apt-get install -y build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python-openssl

- PRA DRIBLAR O PROBLEMA DO OPENSSL:
sudo apt install libssl1.0-dev
sudo apt-get autoremove

- AGR SIM ISNTALA O PYENV:
curl https://pyenv.run | bash

- TEM QUE COLOCAR ISSO AQUI NO BASH.SH:
sudo nano ~/.bashrc
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
sudo reboot now

- O NUMBA DEPENDE DO PYTHON 3.6 PRA CIMA:
pyenv install --list
pyenv install 3.6.1

- CRIA UM AMBIENTE VIRTUAL:
pyenv virtualenv 3.6.1 venv1
pra ativar -> pyenv activate venv1
pra sair -> pyenv deactivate

- SE TENTAR INSALAR O PYAUDIO SEM ISSO AQUI NAO VAI ROLAR:
sudo apt-get install portaudio19-dev python-pyaudio python3-pyaudio
pip install PyAudio

- PRA INSTALAR AS PRÓXIMAS BIBLIOTECAS VAI PRECISAR DISSO AQUI ANTES:
sudo apt-get install libblas-dev liblapack-dev libatlas-base-dev gfortran

- TEM QUE SEGUIR ESSA ORDEM POR CAUSA DE ALGUMAS DEPENDÊNCIAS:
pip install cython
pip install numpy

- TEM QUE INSTALAR UMA VERSAO DO LLVMLITE COMPATIVEL COM O LLVM DO SISTEMA: 
pra encontrar uma versão compatível -> https://pypi.org/project/llvmlite/#description
pip install llvmlite==0.32.0

- TEM QUE ESPECIFICAR A VERSAO DO NUMBA QUE FUNCIONA COM O LLVMLITE 0.32: 
pra encontrar uma versão compatível (ctrl+f llvm) -> https://numba.pydata.org/numba-doc/latest/release-notes.html
pip install numba==0.49.0

- AGORA SIM. AS OUTRAS DEPENDENCIAS O LIBROSA CONSEGUE SOZINHO
pip install librosa
