@echo off
cd /d "%~dp0"
python -m streamlit run main.py --server.port 8501 --server.headless true --server.address 127.0.0.1 > streamlit.out.log 2> streamlit.err.log
