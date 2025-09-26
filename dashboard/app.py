# Tambahkan di bagian sidebar dashboard app.py

# TESTING SECTION
st.sidebar.markdown("---")
st.sidebar.header("🧪 TESTING & DEVELOPMENT")

if st.sidebar.button("Generate Dummy Data"):
    from src.dummy_data_generator import ForexDataGenerator
    generator = ForexDataGenerator()
    generator.save_dummy_data()
    st.sidebar.success("✅ Dummy data generated!")

if st.sidebar.button("Run Quick Backtest"):
    from src.backtester import quick_backtest_test
    results = quick_backtest_test()
    st.sidebar.write("Backtest Results:", results)

if st.sidebar.button("Run System Tests"):
    from tests.test_technical_analysis import run_all_tests
    if run_all_tests():
        st.sidebar.success("✅ All tests passed!")
    else:
        st.sidebar.error("❌ Some tests failed!")
