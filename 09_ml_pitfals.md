# Modul Pembelajaran: Common Machine Learning Pitfalls

Dokumentasi pembelajaran ini membedah berbagai **jebakan dan kesalahan fatal (*pitfalls*)** yang paling sering menjerumuskan perekayasa *Machine Learning* (ML) di dunia nyata. 

Banyak model terlihat sangat cerdas saat diuji di komputer pengembang (*Jupyter Notebook*) dengan angka performa mencapai $99\%$, namun ketika dirilis ke lingkungan operasional (*production*), sistem tersebut gagal total dan menimbulkan kerugian finansial. Modul ini membahas empat jebakan terbesar: **kebocoran data**, **evaluasi yang keliru**, **mengabaikan model acuan dasar**, dan **mempercayai metrik evaluasi secara buta**.

---

## Glosarium Istilah Penting
Pahami beberapa istilah kunci berikut sebelum mendalami materi:
* **Jebakan Fatal (*Pitfall*)**: kesalahan dalam metodologi, asumsi, atau alur kerja pengolahan data yang menghasilkan model rusak atau kesimpulan evaluasi palsu.
* **Kebocoran Data (*Data Leakage*)**: situasi di mana informasi yang seharusnya tidak diketahui oleh model di dunia nyata ikut merembes ke dalam proses latihan.
* **Model Acuan Dasar (*Baseline Model*)**: model pembanding paling sederhana (seperti tebakan rata-rata atau model linear polos) yang dijadikan patokan standar minimal kinerja.
* **Bias Optimisme Semu (*Optimism Bias*)**: ilusi bahwa model bekerja sangat hebat hanya karena model diuji pada data yang sudah dihafal atau dibantu petunjuk tersembunyi.
* **Data Timpang (*Imbalanced Data*)**: kondisi di mana salah satu kelompok target berjumlah sangat mendominasi (misalnya $99,5\%$ transaksi normal vs $0,5\%$ penipuan).
* **Hukum Goodhart (*Goodhart's Law*)**: prinsip sosial-komputasi yang menyatakan bahwa ketika suatu indikator ukuran dijadikan sasaran utama, indikator tersebut akan kehilangan fungsinya sebagai alat ukur yang baik karena sistem cenderung bermain curang untuk memuaskannya.

---

## 1. Peta Besar Jebakan Fatal dalam Machine Learning

Di lingkungan industri, kegagalan proyek Machine Learning jarang sekali disebabkan oleh ketidakmampuan algoritma dalam mengolah data. Sebagian besar kegagalan berakar dari kecerobohan metodologi kerja manusia:

```
+-----------------------------------------------------------------------------+
|                     EMPAT JEBAKAN UTAMA DALAM REKAYASA ML                   |
+-----------------------------------------------------------------------------+

  1. DATA LEAKAGE              2. IMPROPER EVALUATION
     (Kebocoran Informasi)        (Evaluasi Cacat Metodologi)
     - Model membaca contekan.    - Uji acak pada data waktu (time series).
     - Sukses palsu di laptop,    - Membocorkan data uji berulang kali.
       hancur di dunia nyata.      - Optimisme semu pada data latih.
                 \                      /
                  \                    /
                   v                  v
         +--------------------------------------+
         | MODEL GAGAL DI LINGKUNGAN PRODUKSI   |
         | (Sistem Rugi Biaya, Waktu, & Reputasi)|
         +--------------------------------------+
                   ^                  ^
                  /                    \
                 /                      \
  3. IGNORING BASELINES        4. BLINDLY TRUSTING METRICS
     (Mengabaikan Acuan Dasar)    (Percaya Metrik Secara Buta)
     - Langsung pakai model rumit - Terjebak akurasi 99% pada data timpang.
       tanpa menguji logika simpel.- Mengabaikan analisis irisan kelompok.
     - Biaya server membengkak.   - Metrik offline bagus, bisnis nyata anjlok.
```

---

## 2. Kebocoran Data (*Data Leakage*)

Kebocoran data adalah salah satu bentuk kesalahan paling berbahaya karena sering kali tidak memunculkan pesan galat (*error message*) apa pun pada kode program. Model justru akan terlihat memiliki skor yang luar biasa tinggi saat dilatih.

### 2.1 Definisi Konseptual
**Kebocoran Data** terjadi ketika fitur-fitur masukan yang digunakan untuk melatih model memuat informasi rahasia yang **hanya ada di masa depan** atau **tidak akan tersedia saat model digunakan di dunia nyata**.

* **Analogi Pasien Rumah Sakit**: bayangkan kamu ingin membuat model untuk mendeteksi apakah seorang pasien di unit gawat darurat menderita penyakit radang usus buntu atau tidak.
  * Di dalam tabel data historis, terdapat kolom bernama `Nomor Ruang Operasi Bedah`.
  * Model menemukan korelasi sempurna: siapa pun yang memiliki nomor ruang bedah pasti divonis radang usus buntu ($100\%$ akurat).
  * **Masalahnya**: di dunia nyata, nomor ruang operasi baru dicatat oleh perawat **setelah** dokter memutuskan pasien harus dioperasi! Saat ada pasien baru datang pertama kali ke ruang UGD, kolom ini belum terisi, sehingga model langsung kehilangan arah dan tidak bisa bekerja.

---

### 2.2 Tiga Jalur Utama Terjadinya Kebocoran Data

```
                             JALUR KEBOCORAN DATA
                                      |
         +----------------------------+----------------------------+
         |                                                         |
         v                                                         v
  TARGET LEAKAGE                                            CONTAMINATION LEAKAGE
 (Kebocoran dari Masa Depan)                               (Pencemaran Latih-Uji)
- Menggunakan fitur yang baru ada                          - Menghitung normalisasi/imputasi
  setelah kejadian selesai.                                  pada seluruh dataset sebelum split.
- Contoh: Status klaim asuransi                            - Informasi data uji merembes
  dipakai memprediksi risiko kecelakaan.                     ke dalam data latih via rata-rata.
```

#### A. Kebocoran Target dari Masa Depan (*Target Leakage*)
Menggunakan kolom fitur yang secara tidak sadar merupakan akibat atau konsekuensi langsung dari target yang ingin ditebak:
* *Contoh Finansial*: memprediksi apakah seorang debitur akan gagal bayar pinjaman (*default*) menggunakan kolom `Jumlah Pembayaran Denda Keterlambatan`. Seseorang baru membayar denda keterlambatan jika ia memang sudah terbukti terlambat membayar pinjamannya.

#### B. Pencemaran Pra-Pemrosesan (*Preprocessing Contamination*)
Melakukan manipulasi pembersihan data sebelum data dipotong menjadi data latih dan data uji:
* Menghitung nilai rata-rata (*mean*) untuk mengisi nilai kosong pada seluruh dataset secara sekaligus.
* Melakukan penyesuaian rentang skala (*Min-Max Scaling*) menggunakan nilai minimum dan maksimum dari seluruh dataset.
* **Dampak**: data uji telah menyuntikkan informasi sebaran statistiknya ke dalam data latih.

#### C. Kebocoran Duplikasi Subjek (*Group Leakage*)
Mengacak data (*random split*) tanpa memperhatikan bahwa beberapa baris data berasal dari orang, perangkat, atau sesi yang sama:
* Satu pengguna merekam suara kata "Halo" sebanyak $20$ kali.
* Jika diacak, $15$ rekaman masuk ke data latih dan $5$ rekaman masuk ke data uji.
* Model tidak belajar mengenali arti kata "Halo", melainkan menghafal karakteristik frekuensi nada suara unik dari orang tersebut.

---

### 2.3 Cara Menemukan dan Mencegah Kebocoran Data
1. **Gunakan Jalur Pipa Otomatis (*Pipelines*)**: di pustaka seperti Scikit-Learn, gunakan fungsi `Pipeline`. Pipeline memastikan seluruh proses pembersihan (seperti `StandardScaler` atau `SimpleImputer`) hanya dipelajari (*fit*) dari data latih, lalu diterapkan (*transform*) secara terpisah ke data uji.
2. **Pisahkan Data Berdasarkan Garis Waktu (*Temporal Split*)**: untuk data yang terikat waktu, selalu latih model dengan data masa lalu dan uji dengan data masa depan.
3. **Curigai Akurasi yang Terlalu Sempurna**: jika modelmu langsung mendapatkan akurasi di atas $99\%$ pada percobaan pertama untuk masalah yang rumit, hampir dapat dipastikan terjadi kebocoran data.

---

### 2.4 Sumber Rujukan Topik 2
* **Kaufman, Shachar, Rosset, Saharon, dan Perlich, Claudia (2012)**, “Leakage in Data Mining: Formulation, Detection, and Avoidance”, *ACM Transactions on Knowledge Discovery from Data (TKDD)*, 6(4), hlm. 1–21.
* **Géron, Aurélien (2022)**, *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*, Edisi ke-3, O'Reilly Media, Bab 2: “Prepare the Data for Machine Learning Algorithms (Pipelines)”.

---

## 3. Evaluasi yang Keliru (*Improper Evaluation*)

Mengevaluasi model dengan metode yang salah akan melahirkan **optimisme semu**—keyakinan palsu bahwa sistem bekerja sangat baik padahal fondasi pengujiannya cacat.

---

### 3.1 Tiga Bentuk Kesalahan Evaluasi yang Paling Sering Terjadi

#### A. Mengevaluasi Model pada Data Latih (*Resubstitution Error*)
Mengukur kepintaran model menggunakan data yang sama dengan data yang dipakai saat belajar.
* Menghitung akurasi di data latih hanya membuktikan apakah model bisa **menghafal**, bukan apakah model bisa **melakukan generalisasi**.
* Model kompleks seperti *Random Forest* atau *Deep Neural Networks* bisa dengan mudah meraih akurasi $100\%$ pada data latih sambil gagal total menebak data baru (*overfitting* parah).

#### B. Mengacak Data Waktu Secara Bebas (*Random Shuffle on Time Series*)
Memotong data deret waktu menggunakan fungsi pengacak baris (*random train-test split*).
* **Mengapa Keliru?**: data di alam semesta berjalan satu arah secara kronologis. Masa lalu memengaruhi masa depan, bukan sebaliknya.
* Jika data diacak, model menggunakan data hari Jumat dan Rabu untuk memprediksi harga saham hari Kamis. Model secara curang menggunakan informasi masa depan untuk menebak masa lalu.

```
PENGUJIAN DERET WAKTU:

CARA SALAH (Mengacak Data):
[ Hari 1 ] [ Hari 4 ] [ Hari 2 ] ---> Data Latih
[ Hari 3 ] [ Hari 5 ]            ---> Data Uji (Model curang melihat masa depan!)

CARA BENAR (Potongan Kronologis):
[ Bulan Jan - Okt ] ---> Data Latih
[ Bulan Nov       ] ---> Data Validasi
[ Bulan Des       ] ---> Data Uji (Hanya boleh menebak ke depan)
```

#### C. Mengotori Data Uji Berulang Kali (*Data Snooping Bias*)
Membuka dan menggunakan data uji untuk menyetel parameter luar (*hyperparameters*).
* Data uji (*test set*) **hanya boleh disentuh tepat satu kali** di ujung proyek untuk penilaian akhir.
* Jika kamu melihat hasil data uji jelek, lalu kamu mengubah-ubah setelan kedalaman pohon atau laju belajar berulang kali agar skor data uji tersebut naik, maka data uji tersebut kini telah **terkontaminasi oleh keputusan manualmu**. Data tersebut bukan lagi penguji independen yang objektif.

---

### 3.2 Sumber Rujukan Topik 3
* **Hastie, Trevor, Tibshirani, Robert, dan Friedman, Jerome (2009)**, *The Elements of Statistical Learning*, Edisi ke-2, Springer, Bab 7: “Model Assessment and Selection”.
* **Kohavi, Ron (1995)**, “A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection”, *International Joint Conference on Artificial Intelligence (IJCAI)*, 14(2), hlm. 1137–1145.

---

## 4. Mengabaikan Model Acuan Dasar (*Ignoring Baseline Models*)

Banyak praktisi yang baru mempelajari machine learning langsung tergoda untuk menggunakan algoritma paling canggih dan rumit (seperti *XGBoost*, jaringan saraf konvolusi, atau arsitektur Transformer) tanpa pernah membangun model acuan dasar (*baseline*).

---

### 4.1 Apa Risiko Melewatkan Model Acuan Dasar?
1. **Tidak Tahu Apakah Kompleksitas Itu Berguna**:
   * Jika model *Deep Learning* membutuhkan biaya server $\text{Rp}50.000.000$ per bulan dan waktu komputasi 5 jam untuk menghasilkan akurasi $88\%$, sedangkan aturan logika sederhana menghasilkan akurasi $87,5\%$, maka penggunaan *Deep Learning* adalah pemborosan sumber daya organisasi.
2. **Kehilangan Sinyal Peringatan Kerusakan Data**:
   * Model acuan dasar adalah alat diagnosa paling cepat. Jika model regresi linear yang sangat sederhana gagal menghasilkan skor di atas tebakan acak, kemungkinan besar data masukanmu memang tidak memiliki korelasi apa pun dengan target yang ingin ditebak.

---

### 4.2 Dua Tingkatan Baseline Wajib

```
                          DUA TINGKATAN MODEL ACUAN DASAR
                                         |
         +-------------------------------+-------------------------------+
         |                                                               |
         v                                                               v
  NAIVE / HEURISTIC BASELINE                                      SIMPLE ML BASELINE
 (Tingkat Nol Tanpa Algoritma Cerdas)                            (Tingkat Satu Algoritma Ringan)
- Regresi    : Selalu tebak nilai rata-rata.                     - Regresi   : Regresi Linear.
- Klasifikasi: Selalu tebak kelas mayoritas.                     - Klasifikasi: Regresi Logistik / Pohon Tunggal.
- Bisnis     : Aturan logika manual IF-ELSE manusia.             - NLP       : Naïve Bayes + TF-IDF.
```

Sebelum mengklaim model cerdasmu berhasil, kamu harus membuktikan bahwa modelmu mampu mengalahkan **Naive Baseline** dan **Simple ML Baseline** dengan selisih yang signifikan.

---

### 4.3 Sumber Rujukan Topik 4
* **Zinkevich, Martin**, “Rules of Machine Learning: Best Practices for ML Engineering”, Google Research, Aturan #1: “Don't be afraid to launch a product without machine learning” dan Aturan #4: “Keep the first model simple and get the infrastructure right”.
* **Huyen, Chip (2022)**, *Designing Machine Learning Systems*, O'Reilly Media, Bab 6: “Model Development and Offline Evaluation (Baselines)”.

---

## 5. Mempercayai Metrik Secara Buta (*Blindly Trusting Metrics*)

Angka metrik di layar monitor komputer dapat memberikan rasa aman palsu jika praktisi tidak memahami konteks di balik perhitungannya.

---

### 5.1 Paradoks Akurasi pada Kasus Data Timpang
Mengandalkan persentase akurasi murni pada dataset yang distribusinya timpang adalah salah satu jebakan paling klasik:

* **Skenario Deteksi Pembobolan Rekening Bank**:
  * Dari $100.000$ transaksi harian, terdapat $99.900$ transaksi normal dan $100$ transaksi penipuan ($0,1\%$).
  * Sebuah model pemalas dibuat dengan aturan sederhana: **selalu tebak bahwa semua transaksi adalah normal**.
  * **Hasil Evaluasi**:
    * Jumlah tebakan benar: $99.900$ transaksi.
    * Akurasi: $\frac{99.900}{100.000} = 99,9\%$.
  * Di atas kertas, akurasi $99,9\%$ terlihat sempurna. Namun dalam kenyataannya, **$100\%$ pembobolan bank lolos tanpa terdeteksi satu pun**, menyebabkan kerugian nasabah.

> **Peringatan**: jangan pernah menggunakan Akurasi sebagai metrik penentu tunggal jika perbandingan jumlah antarkelas tidak seimbang. Gunakan **Precision**, **Recall**, atau **F1-Score**.

---

### 5.2 Hukum Goodhart dalam Evaluasi Model
> *“Ketika suatu indikator pengukuran dijadikan target sasaran utama, indikator tersebut kehilangan kemampuannya sebagai alat ukur yang baik.”*

Jika seorang perekayasa ML hanya difokuskan untuk memaksimalkan satu metrik tertentu (misalnya tingkat klik iklan / *Click-Through Rate*):
* Model akan belajar menyodorkan judul-judul jebakan sensasional palsu (*clickbait*) karena model menyadari trik tersebut menaikkan rasio klik dengan cepat.
* **Akibatnya**: metrik klik naik drastis di layar metrik, namun kepuasan pengguna jatuh dan reputasi platform hancur di dunia nyata.

---

### 5.3 Jebakan Nilai Rata-Rata Agregat (*The Flaw of Averages*)
Melihat performa model hanya dari satu nilai rata-rata keseluruhan dapat menyembunyikan kegagalan fatal pada kelompok-kelompok tertentu.

* Sebuah model pengenalan wajah memiliki akurasi global $96\%$.
* Ketika dilakukan analisis irisan data (*sliced analysis*):
  * Akurasi pada pria kulit terang: $99,5\%$.
  * Akurasi pada wanita kulit gelap: $65,0\%$.
* Nilai rata-rata global yang tinggi menutupi fakta bahwa model memiliki bias diskriminasi yang sangat parah dan berbahaya terhadap kelompok demografi tertentu.

---

### 5.4 Sumber Rujukan Topik 5
* **Powers, David M. W. (2011)**, “Evaluation: From Precision, Recall and F-Measure to ROC, Informedness, Markedness & Correlation”, *Journal of Machine Learning Technologies*, 2(1), hlm. 37–63.
* **Sculley, David, dkk. (2015)**, “Hidden Technical Debt in Machine Learning Systems”, *Advances in Neural Information Processing Systems (NeurIPS)*, 28, hlm. 2503–2511.
* **Buolamwini, Joy, dan Gebru, Timnit (2018)**, “Gender Shades: Intersectional Accuracy Disparities in Commercial Gender Classification”, *Proceedings of Machine Learning Research*, 81, hlm. 1–15.

---

## 6. Lembar Periksa Keselamatan Rekayasa ML (*Safety Checklist*)

Sebelum memutuskan bahwa sebuah model layak dirilis ke peladen produksi, jalankan pemeriksaan pada tabel evaluasi berikut:

| Area Pemeriksaan | Pertanyaan Penguji Keamanan | Status Verifikasi |
| :--- | :--- | :--- |
| **Kebocoran Data** | Apakah semua fitur masukan dijamin sudah tersedia saat pengguna membuat permintaan inferensi di produksi? | Wajib Lolos |
| **Jalur Pipa Data** | Apakah proses penskalaan (*scaling*) dan pembersihan data dipelajari secara eksklusif hanya dari data latih? | Wajib Lolos |
| **Garis Waktu** | Jika data berupa deret waktu, apakah data uji dipotong secara kronologis tanpa adanya pengacakan silang masa depan? | Wajib Lolos |
| **Model Acuan** | Apakah model rumit ini terbukti mengalahkan akurasi tebakan rata-rata (*naive*) dan model linear sederhana? | Wajib Lolos |
| **Keseimbangan Kelas** | Jika target langka ($<5\%$), apakah kita sudah memeriksa nilai *Recall*, *Precision*, atau *PR-AUC* alih-alih akurasi? | Wajib Lolos |
| **Analisis Irisan** | Apakah performa model sudah diuji secara terpisah pada kelompok-kelompok minoritas penting? | Wajib Lolos |
| **Kerahasiaan Uji** | Apakah data uji (*test set*) hanya disentuh satu kali saja dan tidak dijadikan bahan eksperimen coba-coba? | Wajib Lolos |

---

## 7. Daftar Pustaka Terverifikasi

1. **Kaufman, Shachar, Rosset, Saharon, dan Perlich, Claudia (2012)**. “Leakage in Data Mining: Formulation, Detection, and Avoidance”. *ACM Transactions on Knowledge Discovery from Data (TKDD)*, 6(4), hlm. 1–21.
2. **Hastie, Trevor, Tibshirani, Robert, dan Friedman, Jerome (2009)**. *The Elements of Statistical Learning: Data Mining, Inference, and Prediction*, Edisi ke-2. Springer Science & Business Media.
3. **Zinkevich, Martin**. “Rules of Machine Learning: Best Practices for ML Engineering”. Google Research.
4. **Géron, Aurélien (2022)**. *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*, Edisi ke-3. O'Reilly Media.
5. **Huyen, Chip (2022)**. *Designing Machine Learning Systems*. O'Reilly Media.
6. **Kohavi, Ron (1995)**. “A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection”. *International Joint Conference on Artificial Intelligence (IJCAI)*, 14(2), hlm. 1137–1145.
7. **Powers, David M. W. (2011)**. “Evaluation: From Precision, Recall and F-Measure to ROC, Informedness, Markedness & Correlation”. *Journal of Machine Learning Technologies*, 2(1), hlm. 37–63.
8. **Sculley, David, dkk. (2015)**. “Hidden Technical Debt in Machine Learning Systems”. *Advances in Neural Information Processing Systems (NeurIPS)*, 28, hlm. 2503–2511.
9. **Buolamwini, Joy, dan Gebru, Timnit (2018)**. “Gender Shades: Intersectional Accuracy Disparities in Commercial Gender Classification”. *Proceedings of Machine Learning Research (PMLR)*, 81, hlm. 1–15.