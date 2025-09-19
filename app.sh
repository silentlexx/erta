#!/bin/bash

install(){
    pip install streamlit pandas matplotlib folium
}

if [[ "$1" == "-i" ]]; then
    install
    exit 0
fi

streamlit run app.py
