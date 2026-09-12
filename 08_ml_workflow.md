# Modul Pembelajaran: End-to-End Machine Learning Workflow

Modul pembelajaran ini membedah siklus hidup lengkap (*lifecycle*) rekayasa *Machine Learning* (ML) dari hulu ke hilir: **bagaimana sebuah proyek ML dibangun secara terstruktur mulai dari merumuskan masalah bisnis, mengumpulkan data, eksperimen model, hingga kesadaran operasional saat model dilepas ke peladen produksi (*production*)**.

Banyak pemula mengira pekerjaan Machine Learning hanya seputar memanggil fungsi algoritma di *Jupyter Notebook*. Padahal, penulisan kode algoritma hanyalah sebagian kecil dari keseluruhan rantai kerja rekayasa sistem yang utuh.

---

## Glosarium Istilah Penting
Pahami istilah-istilah kunci berikut sebelum mempelajari tahapan alur kerja:
* **Alur Kerja Ujung-ke-Ujung (*End-to-End Workflow*)**: rangkaian proses terpadu dan berurutan dari awal perumusan masalah sampai sistem terpasang dan dipantau di dunia nyata.
* **Jalur Pipa Data (*Data Pipeline*)**: serangkaian langkah otomatis untuk menyedot, membersihkan, mengubah bentuk, dan mengirimkan data dari sumber mentah ke model.
* **Analisis Data Eksploratif (*Exploratory Data Analysis / EDA*)**: proses investigasi awal pada data untuk memahami sebaran nilai, mendeteksi keanehan, dan menemukan pola menarik sebelum model dibuat.
* **Rekayasa Fitur (*Feature Engineering*)**: proses menciptakan kolom informasi baru dari data mentah agar algoritma lebih mudah menangkap pola (misalnya: mengubah tanggal lahir menjadi kolom usia).
* **Penyetelan Tombol Luar (*Hyperparameter Tuning*)**: eksperimen mencari kombinasi konfigurasi terbaik di luar model (seperti kedalaman pohon atau laju belajar).
* **Penyebaran / Rilis Sistem (*Deployment*)**: proses menanamkan berkas model yang sudah terlatih ke dalam server atau aplikasi agar bisa melayani tebakan untuk pengguna nyata.
* **Pergeseran Data (*Data Drift / Concept Drift*)**: fenomena penurunan performa model seiring berjalannya waktu karena karakteristik data dunia nyata berubah dibanding saat model pertama kali dilatih.

---

## 1. Peta Besar Alur Kerja Machine Learning

Alur kerja Machine Learning bukanlah garis lurus satu arah, melainkan **proses iteratif yang berulang (*iterative loop*)**. Setiap kali menemukan kelemahan di tahap pengujian, kita sering kali harus kembali ke tahap pembersihan data atau bahkan merumuskan ulang pertanyaan bisnisnya.

Standar industri modern mengadaptasi kerangka kerja seperti **CRISP-DM** (*Cross-Industry Standard Process for Data Mining*) dan prinsip rekayasa **MLOps** (*Machine Learning Operations*):

```
+-----------------------------------------------------------------------------+
|                     SIKLUS HIDUP MACHINE LEARNING END-TO-END                |
+-----------------------------------------------------------------------------+

  [ 1. Definisi Masalah ] <---+
            |                 |
            v                 |  Iterasi &
  [ 2. Pengumpulan Data ]     |  Evaluasi Ulang
            |                 |  Berkala
            v                 |
  [ 3. Pemahaman Data (EDA) ] |
            |                 |
            v                 |
  [ 4. Siklus Latih-Validasi ]+
            |
            v
  [ 5. Pemilihan Model Terbaik ]
            |
            v
  [ 6. Evaluasi Akhir (Test Set) ]
            |
            v
  [ 7. Rilis & Pemantauan Produksi ] ---> Terjadi Drift? ---> Kembali ke Tahap 2!
```

---

## 2. Tahap 1: Definisi Masalah (*Problem Definition*)

Tahap pertama dan paling kritis adalah memastikan bahwa Machine Learning memang merupakan solusi yang tepat untuk masalah bisnis yang dihadapi.

### 2.1 Menerjemahkan Sasaran Bisnis Menjadi Tugas ML
Masalah bisnis nyata tidak pernah datang dalam format persamaan matematika. Tugas perekayasa ML adalah menjembatani kebutuhan bisnis tersebut:
* **Masalah Bisnis**: *"Banyak nasabah bank yang tiba-tiba menutup rekening dan pindah ke bank lain (*churn*). Kita ingin mencegah hal itu."*
* **Terjemahan Tugas ML**: *"Membangun sistem klasifikasi biner untuk memprediksi probabilitas seorang nasabah akan berhenti berlangganan dalam 30 hari ke depan berdasarkan riwayat transaksinya."*

---

### 2.2 Menyelaraskan Metrik Bisnis dengan Metrik ML
Model dengan skor akurasi $95\%$ tidak ada artinya jika tidak memberikan dampak finansial positif bagi organisasi:
* **Metrik ML**: memaksimalkan nilai *Recall* hingga di atas $85\%$ (agar nasabah yang berniat pindah tidak banyak yang luput).
* **Metrik Bisnis**: menurunkan angka kehilangan nasabah sebesar $10\%$ dan menekan biaya promosi retensi agar laba bersih meningkat.

---

### 2.3 Mengidentifikasi Kendala Sistem Sejak Awal
Sebelum menulis satu baris kode pun, tentukan batasan operasional sistem:
1. **Batas Waktu Respon (*Latency Constraint*)**: apakah model harus menebak dalam hitungan milidetik secara langsung (misal: otorisasi gesek kartu kredit), atau boleh diproses santai setiap tengah malam (*batch processing*)?
2. **Kebutuhan Penjelasan (*Explainability*)**: apakah model wajib menjelaskan alasan di balik keputusannya kepada regulator hukum (seperti penolakan kredit pemilikan rumah), ataukah model boleh berbentuk kotak hitam (*black-box*)?
3. **Ketersediaan Komputasi dan Anggaran**: apakah sistem akan dijalankan di komputer awan (*cloud*) berbayar mahal atau harus muat di memori ponsel pintar yang terbatas (*edge device*)?

---

### 2.4 Sumber Rujukan Tahap 1
* **Zinkevich, Martin**, “Rules of Machine Learning: Best Practices for ML Engineering”, Google Research, Aturan #1: “Don't be afraid to launch a product without machine learning”.
* **Huyen, Chip (2022)**, *Designing Machine Learning Systems*, O'Reilly Media, Bab 2: “Machine Learning Systems Design”.

---

## 3. Tahap 2: Pengumpulan Data (*Data Collection*)

Model yang canggih tidak akan pernah bisa menyelamatkan data yang berkualitas buruk (*Garbage In, Garbage Out*).

```
                            SUMBER-SUMBER DATA MENTAH
                                       |
        +------------------------------+------------------------------+
        |                              |                              |
        v                              v                              v
  BASIS DATA INTERNAL            LOG SISTEM / IOT              SUMBER EKSTERNAL
 (Tabel SQL, Transaksi,       (Rekaman Klik Web,             (API Pihak Ketiga,
  Data Profil Pengguna)        Sensor Suhu, Telemetri)        Web Scraping, Open Data)
```

---

### 3.1 Aspek Kritis dalam Pengumpulan Data
1. **Keterwakilan Sampel (*Representativeness*)**:
   * Data latih yang dikumpulkan harus mencerminkan kondisi nyata di masa depan. Jika kamu ingin memprediksi harga rumah di seluruh Indonesia, kamu tidak boleh hanya mengumpulkan data rumah di Jakarta Selatan.
2. **Bias Pengumpulan (*Selection Bias*)**:
   * Hati-hati terhadap data yang hanya tercatat karena kondisi tertentu (misal: data survei online hanya diisi oleh anak muda yang melek internet, mengabaikan populasi lansia).
3. **Kepatuhan Privasi dan Regulasi Hukum**:
   * Pengumpulan data pengguna wajib mematuhi regulasi perlindungan data pribadi (seperti UU Perlindungan Data Pribadi / PDP di Indonesia atau GDPR di Eropa). Informasi sensitif seperti nomor identitas kependudukan, kata sandi, dan data rekam medis harus disamarkan (*anonymized*).

---

### 3.2 Sumber Rujukan Tahap 2
* **Sambasivan, Nithya, dkk. (2021)**, “Everyone wants to do the model work, not the data work: Data Cascades in High-Stakes AI”, *Proceedings of the 2021 CHI Conference on Human Factors in Computing Systems*, hlm. 1–15.
* **Gudivada, Venkat, dkk. (2017)**, “Data Quality Considerations for Big Data and Machine Learning Applications”, *Data Science Journal*, 16, hlm. 1–27.

---

## 4. Tahap 3: Pemahaman Data (*Data Understanding & EDA*)

Setelah data mentah terkumpul, jangan langsung menyodorkannya ke algoritma. Kamu harus melakukan investigasi mendalam melalui **Analisis Data Eksploratif (*Exploratory Data Analysis / EDA*)**.

```
                        CHECKLIST PEMAHAMAN DATA (EDA)
                                       |
    +------------------+------------------+------------------+------------------+
    |                  |                  |                  |                  |
    v                  v                  v                  v                  v
Pemeriksaan Tipe   Pemeriksaan        Deteksi Titik      Analisis Sebaran   Pemeriksaan
Kolom & Format     Nilai Kosong       Pencilan Liar      & Hubungan Korelasi Kebocoran Data
(Teks vs Angka)    (Missing Values)   (Outliers)         (Skewness / Matrix) (Data Leakage)
```

---

### 4.1 Langkah Operasional Pemahaman Data
1. **Analisis Statistik Ringkas**:
   * Periksa nilai rata-rata (*mean*), nilai tengah (*median*), nilai minimum, maksimum, dan simpangan baku (*standard deviation*) pada setiap kolom fitur.
2. **Pemeriksaan Nilai Kosong (*Missing Values*)**:
   * Cari tahu kolom mana yang bolong dan mengapa data tersebut bisa hilang (apakah sensor rusak, pengguna sengaja melewati pertanyaan, atau kesalahan sistem pencatatan?).
3. **Analisis Sebaran dan Bentuk Data (*Distribution Profiling*)**:
   * Buat grafik histogram. Apakah datanya simetris membentuk kurva lonceng normal, ataukah miring ekstrem (*skewed*) ke satu sisi?
4. **Deteksi Awal Kebocoran Data (*Leakage Hunting*)**:
   * Periksa apakah ada kolom fitur yang nilainya baru bisa diketahui **setelah** target tebakan terjadi di masa depan. Jika ada, kolom tersebut wajib dibuang segera.

---

### 4.2 Sumber Rujukan Tahap 3
* **Tukey, John W. (1977)**, *Exploratory Data Analysis*, Addison-Wesley Publishing Company.
* **Géron, Aurélien (2022)**, *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*, Edisi ke-3, O'Reilly Media, Bab 2: “End-to-End Machine Learning Project (Discover and Visualize the Data to Gain Insights)”.

---

## 5. Tahap 4: Siklus Latih-Validasi (*Train–Validate Loop*)

Inilah laboratorium eksperimen tempat model dibentuk, diuji coba, dan disetel secara berulang-ulang.

```
+-----------------------------------------------------------------------------+
|                          SIKLUS LATIH-VALIDASI ITERATIF                     |
+-----------------------------------------------------------------------------+

  [ Data Latih ] ---> [ Pra-Pemrosesan & Rekayasa Fitur ]
                             |
                             v
                      [ Latih Model Awal ]
                             |
                             v
  [ Data Validasi ] -> [ Uji Performa ] <---+
                             |              | Belum Optimal?
                             v              | Ubah setelan tombol luar
                      +-------------+       | atau buat fitur baru!
                      | Evaluasi?   | ------+
                      +-------------+
                             |
                             v Sudah Maksimal!
                      [ Lanjut ke Pemilihan Model ]
```

---

### 5.1 Rincian Langkah dalam Siklus
1. **Pra-Pemrosesan Data (*Data Preprocessing*)**:
   * Mengisi nilai yang hilang (*imputation*).
   * Mengubah teks kategori menjadi format angka yang dipahami komputer (*One-Hot Encoding* atau *Ordinal Encoding*).
   * Menyamakan skala angka fitur (*Feature Scaling* lewat standardisasi atau normalisasi).
2. **Rekayasa Fitur (*Feature Engineering*)**:
   * Menggabungkan atau mengekstrak informasi agar pola lebih terlihat jelas (misalnya: membagi kolom `Total Belanja` dengan `Frekuensi Kunjungan` untuk mendapatkan kolom `Rata-rata Belanja per Kunjungan`).
3. **Penyetelan Tombol Luar (*Hyperparameter Tuning*)**:
   * Mencari kombinasi parameter terbaik secara sistematis menggunakan metode:
     * **Grid Search**: mencoba seluruh kombinasi daftar setelan secara manual satu per satu.
     * **Random Search**: memilih kombinasi acak dalam rentang setelan tertentu (jauh lebih cepat dan efisien).
     * **Bayesian Optimization**: metode cerdas yang memperhitungkan hasil uji coba masa lalu untuk menebak setelan berikutnya yang paling menjanjikan.
4. **Pemantauan Gejala Model**:
   * Pantau kurva belajar (*learning curves*). Jika kesalahan data latih kecil tetapi data validasi besar, model terkena penyakit menghafal mati (*overfitting*). Pasang teknik pembatasan (*regularization*) atau pangkas kompleksitas model.

---

### 5.2 Sumber Rujukan Tahap 4
* **Goodfellow, Ian, Bengio, Yoshua, dan Courville, Aaron (2016)**, *Deep Learning*, MIT Press, Bab 5: “Machine Learning Basics (Hyperparameters and Validation Sets)”.
* **Bergstra, James, dan Bengio, Yoshua (2012)**, “Random Search for Hyper-Parameter Optimization”, *Journal of Machine Learning Research (JMLR)*, 13, hlm. 281–305.

---

## 6. Tahap 5: Pemilihan Model (*Model Selection*)

Dalam fase eksperimen, kita biasanya melatih beberapa keluarga algoritma yang berbeda sekaligus (misalnya: Regresi Logistik, *Random Forest*, *LightGBM*, dan Jaringan Saraf Tiruan). Tahap ini bertujuan memilih satu model terbaik untuk diterbangkan ke produksi.

---

### 6.1 Matriks Evaluasi Pertimbangan Model
Model terbaik **bukan selalu** model yang skor akurasinya paling tinggi. Perekayasa ML harus menimbang aspek pertukaran (*tradeoffs*):

| Kriteria Pertimbangan | Model Sederhana (Regresi Linear / Pohon Tunggal) | Model Ansambel / Kompleks (XGBoost / Random Forest) | Model Jaringan Saraf Tiruan (Deep Learning) |
| :--- | :--- | :--- | :--- |
| **Akurasi Tebakan** | Sedang | **Sangat Tinggi** (pada data tabular) | Sangat Tinggi (pada citra / teks) |
| **Kecepatan Inferensi**| **Sangat Cepat** ($< 1$ milidetik) | Cepat hingga sedang | Lambat (butuh akselerator GPU) |
| **Interpretabilitas** | **Sangat Transparan** (mudah diaudit) | Menengah (butuh alat bantu seperti SHAP) | Kotak Hitam (*Black Box*) |
| **Biaya Server / Daya**| **Sangat Murah** | Menengah | Sangat Mahal |
| **Kebutuhan Volume Data**| Berfungsi baik di data kecil | Cukup stabil di berbagai ukuran data | Wajib data berskala masif |

---

### 6.2 Prinsip Pisau Cukur Occam (*Occam's Razor*)
> *"Di antara beberapa hipotesis yang menghasilkan kemampuan prediksi serupa, pilihlah hipotesis yang paling sederhana."*

Jika model pohon keputusan sederhana menghasilkan akurasi $91\%$ dan model jaringan saraf tiruan menghasilkan akurasi $91,5\%$, pilihlah pohon keputusan. Peningkatan akurasi marjinal sebesar $0,5\%$ tidak sebanding dengan kerumitan infrastruktur, biaya server, dan sulitnya penelusuran galat di masa depan.

---

### 6.3 Sumber Rujukan Tahap 5
* **Hastie, Trevor, Tibshirani, Robert, dan Friedman, Jerome (2009)**, *The Elements of Statistical Learning*, Edisi ke-2, Springer, Bab 7: “Model Assessment and Selection”.
* **Breiman, Leo (2001)**, “Statistical Modeling: The Two Cultures”, *Statistical Science*, 16(3), hlm. 199–231.

---

## 7. Tahap 6: Evaluasi Akhir (*Final Evaluation*)

Tahap ini adalah momen pembuktian kebenaran yang tidak memihak sebelum model dilepas ke dunia nyata.

---

### 7.1 Membuka Brankas Data Uji (*Test Set*)
* **Aturan Mutlak**: data uji (*test set*) yang selama ini dikunci rapat di dalam brankas akhirnya dikeluarkan dan disodorkan ke model terbaik terpilih.
* Pengujian ini **hanya dilakukan tepat satu kali**.
* Jika skor pada data uji mendekati skor pada data validasi, kita memiliki bukti ilmiah yang kuat bahwa model memiliki kemampuan **generalisasi** yang baik dan siap dipasang di produksi.

---

### 7.2 Analisis Irisan Kelompok (*Sliced Analysis*)
Jangan hanya melihat satu angka metrik rata-rata keseluruhan. Sering kali model terlihat hebat secara global, tetapi memiliki cacat diskriminasi yang parah pada segmen tertentu.

Lakukan pengujian terpisah pada irisan kelompok (*slices*):
* Bagaimana performa model pada pengguna lansia vs usia muda?
* Bagaimana performa model pada pengguna dari wilayah terpencil dengan koneksi lambat?
* Bagaimana performa model saat akhir pekan vs hari kerja?

Jika model gagal pada kelompok tertentu, sistem berisiko memicu tuntutan hukum atau kerugian reputasi bisnis saat rilis.

---

### 7.3 Sumber Rujukan Tahap 6
* **Zinkevich, Martin**, “Rules of Machine Learning: Best Practices for ML Engineering”, Google Research, Aturan #40: “Keep ensembles simple”.
* **Barocas, Solon, Hardt, Moritz, dan Narayanan, Arvind (2019)**, *Fairness and Machine Learning: Limitations and Opportunities*, MIT Press, Bab 2: “Classification and its harms”.

---

## 8. Tahap 7: Kesadaran Rilis & Produksi (*Deployment Awareness - High-Level*)

Pekerjaan praktisi Machine Learning **belum selesai** saat model berhasil dilatih. Justru di tahap rilis inilah ujian sesungguhnya dimulai.

```
                            EKOSISTEM SISTEM DI SERVER PRODUKSI
                                             |
        +------------------------------------+------------------------------------+
        |                                                                         |
        v                                                                         v
  CARA PENYAJIAN (SERVING)                                                  PEMANTAUAN (MONITORING)
- Prediksi Borongan (Batch Serving):                                      - Pantau Pergeseran Data (Data Drift).
  Hitung jutaan data tiap malam, simpan ke database.                      - Pantau Waktu Respon Server (Latency).
- Prediksi Langsung (Real-Time API):                                      - Pantau Penurunan Akurasi Model.
  Model memproses permintaan REST API < 50ms.                             - Siapkan Tombol Pemulihan (Rollback).
```

---

### 8.1 Dua Pola Penyajian Model ke Pengguna (*Serving Patterns*)
1. **Penyajian Borongan (*Batch Serving / Offline Prediction*)**:
   * Model membaca jutaan data sekaligus pada jadwal rutin (misal: setiap jam 02.00 dini hari).
   * Hasil tebakan disimpan ke dalam basis data. Saat pengguna membuka aplikasi di pagi hari, aplikasi tinggal membaca hasil yang sudah siap saji.
   * *Contoh*: pembuatan daftar rekomendasi playlist mingguan di Spotify.
2. **Penyajian Langsung (*Real-Time Serving / Online Prediction*)**:
   * Model ditanam di balik gerbang antarmuka pemrograman aplikasi (*REST API / gRPC*).
   * Setiap kali pengguna menekan tombol transaksi, aplikasi mengirimkan data fitur detik itu juga dan model wajib membalas hasil tebakan dalam hitungan milidetik.
   * *Contoh*: deteksi penipuan saat kartu kredit digesek di kasir toko.

---

### 8.2 Mengapa Model Mengalami Penurunan Performa? (*Model Decay & Drift*)
Model Machine Learning tidak seperti perangkat lunak biasa yang kodenya tidak berubah. Performa model **pasti akan memburuk seiring waktu** karena dunia nyata terus bergerak dan berubah:
* **Pergeseran Data (*Data Drift*)**: profil masukan berubah (misalnya: kamera pengguna resolusinya semakin tajam, atau inflasi membuat nominal transaksi bergeser naik).
* **Pergeseran Konsep (*Concept Drift*)**: hubungan makna antara fitur dan target berubah (misalnya: gaya bahasa penipuan email berubah mengikuti tren kata baru).

---

### 8.3 Utang Teknis Tersembunyi (*Hidden Technical Debt*)
Dalam publikasi terkenal dari tim peneliti Google (Sculley dkk., 2015), diungkapkan bahwa kode algoritma ML sebenarnya hanya memakan porsi sekitar $5\%$ dari total sistem perangkat lunak di produksi. Sisanya ($95\%$) adalah infrastruktur pendukung:

```
+-----------------------------------------------------------------------------+
|               UTANG TEKNIS SISTEM MACHINE LEARNING DI INDUSTRI              |
+-----------------------------------------------------------------------------+

  [ Pengumpulan Data ]    [ Verifikasi Data ]      [ Ekstraksi Fitur ]
  
        [ Infrastruktur Pemantauan ]   [ KODE ML ]   [ Manajemen Server ]
                                         (Cuma 5%!)
  
  [ Manajemen Sumber Daya ]    [ Jalur Pipa Sajian ]   [ Pengujian Otomatis ]
```

Perekayasa sistem harus menyiapkan:
* **Sistem Alarm Otomatis**: yang membunyikan peringatan jika sebaran data masukan di server menyimpang dari sebaran data saat latihan.
* **Pemicu Latih Ulang (*Retraining Triggers*)**: mekanisme otomatis untuk melatih ulang model menggunakan pasokan data baru secara berkala.
* **Prosedur Pemulihan (*Rollback Strategy*)**: kemampuan untuk mematikan model bermasalah dan mengembalikan sistem ke versi model stabil sebelumnya dalam hitungan detik.

---

### 8.4 Sumber Rujukan Tahap 7
* **Sculley, David, dkk. (2015)**, “Hidden Technical Debt in Machine Learning Systems”, *Advances in Neural Information Processing Systems (NeurIPS)*, 28, hlm. 2503–2511.
* **Huyen, Chip (2022)**, *Designing Machine Learning Systems*, O'Reilly Media, Bab 9: “Continual Learning and Test in Production” dan Bab 10: “Infrastructure and Tooling for MLOps”.

---

## 9. Kerangka Rangkuman Alur Kerja Lengkap

Gunakan lembar panduan ringkas berikut untuk memandu perjalanan proyek ML dari konsep awal hingga beroperasi di peladen produksi:

```
[ TAHAP 1: DEFINISI MASALAH ]
  - Ubah masalah bisnis menjadi format komputasi ML (Regresi / Klasifikasi / dsb).
  - Tentukan metrik evaluasi keberhasilan dan batas waktu respon sistem.

[ TAHAP 2: PENGUMPULAN DATA ]
  - Tarik data dari basis data internal, log sistem, atau sumber eksternal.
  - Pastikan sampel mewakili populasi nyata dan tidak melanggar hukum privasi.

[ TAHAP 3: PEMAHAMAN DATA (EDA) ]
  - Telusuri sebaran data, bersihkan nilai kosong, dan isolasi titik pencilan liar.
  - Cari dan buang potensi kebocoran data (data leakage) sebelum melangkah maju.

[ TAHAP 4: SIKLUS LATIH-VALIDASI ]
  - Pisahkan data latih dan validasi (gunakan k-Fold jika data terbatas).
  - Lakukan rekayasa fitur dan coba berbagai kombinasi setelan tombol luar.

[ TAHAP 5: PEMILIHAN MODEL ]
  - Timbang akurasi vs kecepatan inferensi vs biaya server.
  - Terapkan Pisau Cukur Occam: pilih model tersederhana yang kinerjanya memadai.

[ TAHAP 6: EVALUASI AKHIR ]
  - Buka brankas data uji (test set) dan uji model tepat satu kali saja.
  - Lakukan analisis irisan kelompok untuk menjamin keadilan sistem.

[ TAHAP 7: RILIS & PEMANTAUAN PRODUKSI ]
  - Tanamkan model ke server (Batch Serving atau Real-Time API).
  - Pasang sistem pemantau pergeseran data (drift) dan jalur latih ulang otomatis.
```

---

## 10. Daftar Pustaka Terverifikasi

1. **Zinkevich, Martin**. “Rules of Machine Learning: Best Practices for ML Engineering”. Google Research.
2. **Huyen, Chip (2022)**. *Designing Machine Learning Systems*. O'Reilly Media.
3. **Sculley, David, dkk. (2015)**. “Hidden Technical Debt in Machine Learning Systems”. *Advances in Neural Information Processing Systems (NeurIPS)*, 28, hlm. 2503–2511.
4. **Géron, Aurélien (2022)**. *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*, Edisi ke-3. O'Reilly Media.
5. **Hastie, Trevor, Tibshirani, Robert, dan Friedman, Jerome (2009)**. *The Elements of Statistical Learning: Data Mining, Inference, and Prediction*, Edisi ke-2. Springer Science & Business Media.
6. **Goodfellow, Ian, Bengio, Yoshua, dan Courville, Aaron (2016)**. *Deep Learning*. MIT Press.
7. **Tukey, John W. (1977)**. *Exploratory Data Analysis*. Addison-Wesley.
8. **Sambasivan, Nithya, dkk. (2021)**. “Everyone wants to do the model work, not the data work: Data Cascades in High-Stakes AI”. *Proceedings of the 2021 CHI Conference on Human Factors in Computing Systems*, hlm. 1–15.
9. **Bergstra, James, dan Bengio, Yoshua (2012)**. “Random Search for Hyper-Parameter Optimization”. *Journal of Machine Learning Research (JMLR)*, 13, hlm. 281–305.
10. **Barocas, Solon, Hardt, Moritz, dan Narayanan, Arvind (2019)**. *Fairness and Machine Learning: Limitations and Opportunities*. MIT Press.
11. **Chapman, Pete, dkk. (2000)**. *CRISP-DM 1.0: Step-by-step data mining guide*. The CRISP-DM Consortium.