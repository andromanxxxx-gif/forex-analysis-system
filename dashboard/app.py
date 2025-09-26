import streamlit as st
import pandas as pd
from src.gdrive_manager import GDRIVE_MANAGER
from src.data_fetcher import ForexDataFetcher

def main():
    st.title("FOREX SYSTEM - Google Drive Integrated")
    
    # Initialize managers
    gdrive = GDRIVE_MANAGER
    fetcher = ForexDataFetcher()
    
    # Pair selection
    pairs = ['GBPJPY', 'USDJPY', 'CHFJPY', 'EURJPY', 'EURNZD']
    selected_pair = st.selectbox("Pilih Pair:", pairs)
    
    # Data operations
    if st.button("🔄 Update dari Yahoo Finance"):
        data = fetcher.get_data(selected_pair)
        gdrive.save_data(data, f"{selected_pair}_data.csv")
        st.success("Data tersimpan di Google Drive!")
    
    if st.button("📊 Load dari Google Drive"):
        data = gdrive.load_data(f"{selected_pair}_data.csv")
        if data is not None:
            st.line_chart(data['Close'])
            
if __name__ == "__main__":
    main()
