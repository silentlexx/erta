#!/bin/bash

install(){
    pip install -r requirements.txt
}

if [[ "$1" == "-i" ]]; then
    install
    exit 0
fi

streamlit run app.py
