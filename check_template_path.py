import os

def check_template_path():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    templates_dir = os.path.join(base_dir, "templates")
    index_path = os.path.join(templates_dir, "index.html")

    print("📂 Lokasi folder utama:", base_dir)
    print("📁 Folder templates:", templates_dir)
    print("📄 File index.html:", index_path)

    if not os.path.exists(templates_dir):
        print("❌ Folder 'templates' tidak ditemukan.")
        return

    files = os.listdir(templates_dir)
    print("\n📜 Isi folder templates:")
    for f in files:
        print("   -", f)

    if os.path.exists(index_path):
        print("\n✅ File 'index.html' ditemukan!")
    else:
        print("\n❌ File 'index.html' TIDAK ADA di folder templates.")

if __name__ == "__main__":
    check_template_path()
