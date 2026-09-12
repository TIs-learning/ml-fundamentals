# Modul Pembelajaran: Bias–Variance Tradeoff

Dokumentasi pembelajaran ini membahas hukum tarik-ulur paling mendasar dalam seluruh cabang *Machine Learning* (ML): **Tarik-Ulur Bias dan Varians (*Bias–Variance Tradeoff*)**. Konsep ini menjelaskan mengapa tidak ada model yang bisa sempurna di segala situasi, bagaimana mendiagnosis model yang terlalu bodoh (*underfitting*) atau terlalu menghafal (*overfitting*), serta langkah rekayasa apa yang harus diambil saat performa model bermasalah di dunia nyata.

---

## Glosarium Istilah Penting
Pahami arti istilah kunci berikut agar pembahasan tidak membingungkan:
* **Bias**: kesalahan sistematis yang muncul karena model membuat asumsi yang terlalu sederhana terhadap pola data asli di dunia nyata.
* **Varians (*Variance*)**: tingkat sensitivitas atau kegoyahan model terhadap perubahan kecil pada data latih.
* **Tarik-Ulur (*Tradeoff*)**: kondisi kompromi di mana kita tidak bisa memperbaiki satu sisi tanpa mengorbankan sisi lainnya (jika bias ditekan turun, varians cenderung naik, dan sebaliknya).
* **Kurang Pas (*Underfitting*)**: kondisi di mana model terlalu kaku dan bodoh sehingga gagal menangkap pola penting pada data latih maupun data uji.
* **Kelewat Pas / Menghafal Mati (*Overfitting*)**: kondisi di mana model terlalu pintar menghafal detail dan gangguan (*noise*) pada data latih, tetapi langsung gagal saat menguji data baru.
* **Derau Tak Tereduksi (*Irreducible Error*, $\sigma^2$)**: batas galat alami yang mustahil dihilangkan oleh algoritma apa pun karena keterbatasan data atau ketidakpastian murni di alam semesta.

---

## 1. Intuisi Bias vs Varians (*Bias vs Variance Intuition*)

Untuk memahami bias dan varians, bayangkan sebuah **papan sasaran tembak (target panahan / dartboard)**. Titik tengah (*bullseye*) adalah target kebenaran sejati yang ingin ditebak. Setiap anak panah yang menancap adalah hasil tebakan dari satu model yang dilatih pada kelompok data yang berbeda.

```
ILUSTRASI PAPAN TARGET TEMBAK:

  (A) BIAS RENDAH, VARIANS RENDAH       (B) BIAS RENDAH, VARIANS TINGGI
           +---------+                           +---------+
           |    x    |                           | x       |
           |   xxx   |                           |    x    |
           |    x    |                           |       x |
           +---------+                           +---------+
        (Akurat & Konsisten)                  (Menyebar Lebar / Goyah)

  (C) BIAS TINGGI, VARIANS RENDAH       (D) BIAS TINGGI, VARIANS TINGGI
           +---------+                           +---------+
           | xxxxx   |                           | x       |
           |         |                           |       x |
           |         |                           | x     x |
           +---------+                           +---------+
       (Konsisten tapi Meleset)               (Meleset Jauh & Berantakan)
```

---

### 1.1 Penjelasan Karakteristik

#### A. Bias: Seberapa Kaku Asumsi Modelmu?
* **Model dengan Bias Tinggi**: memiliki prasangka yang terlalu kaku. Model ini mengabaikan sinyal-sinyal penting di dalam data karena memaksakan asumsi sederhananya.
  * *Contoh*: menggunakan penggaris lurus (Regresi Linear) untuk memprediksi kurva yang jelas-jelas berbentuk parabola melengkung.
  * *Hasil*: tebakan konsisten meleset dari sasaran tengah pada data latih maupun data baru.

#### B. Varians: Seberapa Goyah Tebakan Modelmu?
* **Model dengan Varians Tinggi**: sangat labil dan tidak punya pendirian tetap. Model ini terlalu memperhatikan detail kecil dan derau acak pada data latih.
  * *Ciri Khas*: jika kamu mengganti sedikit saja baris data di kumpulan data latih, struktur rumus dan keputusan model akan langsung berubah secara drastis.
  * *Hasil*: tebakannya menyebar liar tak beraturan.

---

### 1.2 Dekomposisi Matematis Galat Total
Secara matematis, kesalahan tebakan total (*Expected Prediction Error*) dari suatu model pada data baru dapat dipecah menjadi tiga komponen:

$$\text{Total Error} = \text{Bias}^2 + \text{Variance} + \sigma^2$$

1. **$\text{Bias}^2$**: seberapa jauh rata-rata tebakan model menyimpang dari target sejati di lapangan.
2. **$\text{Variance}$**: seberapa besar tebakan model bergoyang di sekitar nilai rata-ratanya jika dilatih pada kelompok data yang berbeda.
3. **$\sigma^2$ (Irreducible Error)**: batas bawah galat tak terhindarkan akibat faktor alam atau variabel yang tidak kita catat di tabel fitur. Sehebat apa pun model yang kamu buat, kesalahan tebakannya tidak akan pernah bisa lebih rendah dari nilai $\sigma^2$ ini.

---

### 1.3 Sumber Rujukan Topik 1
* **Geman, Stuart, Bienenstock, Élie, dan Doursat, René (1992)**, “Neural Networks and the Bias/Variance Dilemma”, *Neural Computation*, 4(1), hlm. 1–58.
* **Hastie, Trevor, Tibshirani, Robert, dan Friedman, Jerome (2009)**, *The Elements of Statistical Learning*, Edisi ke-2, Springer, Bab 7: “Model Assessment and Selection (Bias-Variance Decomposition)”.

---

## 2. Kaitan Langsung: Underfitting vs Overfitting

Dilema bias dan varians tercermin langsung dalam dua istilah paling populer di Machine Learning: **Underfitting** dan **Overfitting**.

```
                        SPEKTRUM KAPASITAS MODEL
                                   |
         +-------------------------+-------------------------+
         |                                                   |
         v                                                   v
   BIAS TINGGI                                         VARIANS TINGGI
  (Underfitting)                                       (Overfitting)
- Model terlalu sederhana.                            - Model terlalu kompleks.
- Gagal di Data Latih.                                - Sempurna di Data Latih.
- Gagal di Data Uji.                                  - Hancur di Data Uji.
```

---

### 2.1 Perbandingan Karakteristik

#### A. Kurang Pas (*Underfitting / High Bias*)
* Model tidak memiliki daya tampung (*capacity*) yang cukup untuk memahami pola matematika data.
* **Gejala Utama**:
  * Nilai kesalahan (*loss/error*) pada **data latih sangat tinggi**.
  * Nilai kesalahan pada **data validasi/uji juga sama tingginya**.
* **Analogi Siswa**: murid malas yang tidak membaca buku teks sama sekali, sehingga nilainya jeblok pada latihan harian maupun saat ujian semester.

#### B. Kelewat Pas (*Overfitting / High Variance*)
* Model memiliki daya tampung berlebih dan menghafal seluruh data latihan sampai ke derau-derau kotornya.
* **Gejala Utama**:
  * Nilai kesalahan pada **data latih mendekati nol (hampir sempurna)**.
  * Nilai kesalahan pada **data validasi/uji melonjak tinggi**.
  * Terjadi **jarak pemisah yang sangat lebar (*large generalization gap*)** antara skor data latih dan skor data uji.
* **Analogi Siswa**: murid yang menghafal mati titik-koma naskah soal latihan kemarin. Jika angka pada soal ujian diubah sedikit saja, ia langsung bingung dan salah menjawab.

---

### 2.2 Grafik Visual Bias-Variance vs Tingkat Kesalahan

```
Nilai Kesalahan ^
                |   \                                  /  (Error Data Uji / Validasi)
                |    \      Zone        Zone          /
                |     \  Underfitting  Overfitting   /
                |      \    (Bias)    (Variance)    /
                |       \                          /
                |        \        Titik Optimal   /
                |         \        (Sweet Spot)  /
                |          \            v       /
                |           \-------+----------/
                |            \      |         /
                |             \     |        /
                |              \----+-------/---------   (Error Data Latih)
                |               \___|______/
                +-------------------+-------------------->
                                  Kompleksitas Model
```

* **Di sisi kiri (Model Sederhana)**: Bias sangat mendominasi. Error data latih dan uji sama-sama tinggi.
* **Di sisi kanan (Model Rumit)**: Varians sangat mendominasi. Error data latih terus menurun, namun error data uji berbelok naik tajam.
* **Titik Optimal (*Sweet Spot*)**: titik lembah di tengah di mana kombinasi $\text{Bias}^2 + \text{Variance}$ menghasilkan nilai total error terendah pada data uji.

---

### 2.3 Sumber Rujukan Topik 2
* **James, Gareth, Witten, Daniela, Hastie, Trevor, dan Tibshirani, Robert (2021)**, *An Introduction to Statistical Learning: with Applications in R*, Edisi ke-2, Springer, Bab 2: “Assessing Model Accuracy”.
* **Géron, Aurélien (2022)**, *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*, Edisi ke-3, O'Reilly Media, Bab 4: “Learning Curves”.

---

## 3. Kompleksitas Model vs Ukuran Data (*Model Complexity vs Data Size*)

Dua faktor utama yang mengendalikan posisi model pada spektrum bias-varians adalah **seberapa rumit model tersebut** dan **seberapa banyak data yang kita berikan**.

---

### 3.1 Pengaruh Kompleksitas Model (*Model Complexity*)
Kompleksitas model diukur dari banyaknya parameter dalam yang bebas bergerak (misal: derajat polinomial regresi, kedalaman pohon keputusan, atau jumlah lapisan saraf tiruan).

* **Menaikkan Kompleksitas**:
  * Menurunkan Bias (model lebih leluasa membentuk batas keputusan meliuk).
  * Menaikkan Varians (model semakin sensitif terhadap kebetulan statistik pada sampel data).
* **Menurunkan Kompleksitas**:
  * Menaikkan Bias (model dipaksa mengikuti batas keputusan yang kaku).
  * Menurunkan Varians (model lebih stabil dan tahan banting).

---

### 3.2 Pengaruh Ukuran Data Latih (*Data Size / Sample Size*)
Salah satu cara paling ampuh untuk menekan varians tanpa perlu mengorbankan bias adalah **menambah volume data latih ($N$)**.

```
PENGARUH PENAMBAHAN DATA LATIH:

Dataset Kecil (N Sedikit):
  Satu data pencilan liar dapat membelokkan kurva model secara drastis (Varians Tinggi).

Dataset Raksasa (N Banyak):
  Data pencilan tenggelam oleh jutaan data normal lainnya. Pola sejati dunia nyata
  menjadi lebih dominan daripada derau lokal (Varians Turun!).
```

* **Untuk Model Berkapasitas Tinggi (seperti Jaringan Saraf Tiruan Dalam)**: model ini butuh pasokan data berukuran masif agar variansnya terkendali dan performa puncaknya tercapai.
* **Untuk Model Berbias Tinggi (seperti Regresi Linear Sederhana)**: menambah data dari 100 ribu menjadi 10 juta baris **tidak akan menolong**. Model tersebut sudah mencapai batas kapasitas matematiknya; error-nya akan tetap tinggi karena garis lurus tidak akan pernah bisa mengejar pola melengkung.

---

### 3.3 Sumber Rujukan Topik 3
* **Goodfellow, Ian, Bengio, Yoshua, dan Courville, Aaron (2016)**, *Deep Learning*, MIT Press, Bab 5: “Capacity, Overfitting and Underfitting”.
* **Ng, Andrew (2018)**, *Machine Learning Yearning*, Bab 13: “Diagnosing Bias and Variance: Learning Curves”.

---

## 4. Bagaimana Bias–Variance Muncul di Praktik Kerja?

Di industri nyata, kita tidak bisa melihat rumus matematis bias dan varians secara terpisah. Kita mendiagnosisnya lewat **perbandingan antara skor pada data latih (*train score*) dan skor pada data validasi (*validation score*)**.

---

### 4.1 Tabel Diagnosis Cepat Masalah Performa
Asumsikan kita menargetkan tingkat kesalahan rendah (misalnya batas toleransi galat manusia adalah $1\%$):

| Kasus | Galat Data Latih | Galat Data Validasi | Diagnosis Masalah | Status Model |
| :--- | :--- | :--- | :--- | :--- |
| **Kasus 1** | $15\%$ *(Tinggi)* | $16\%$ *(Tinggi)* | **High Bias** | Model terlalu kaku (*Underfitting*) |
| **Kasus 2** | $1\%$ *(Rendah)* | $12\%$ *(Tinggi)* | **High Variance** | Model menghafal mati (*Overfitting*) |
| **Kasus 3** | $14\%$ *(Tinggi)* | $28\%$ *(Sangat Tinggi)* | **High Bias & High Variance** | Kasus terburuk: kaku sekaligus tidak stabil |
| **Kasus 4** | $1,2\%$ *(Rendah)* | $1,5\%$ *(Rendah)* | **Low Bias & Low Variance** | **Sempurna / Ideal** |

---

### 4.2 Langkah Aksi Praktis untuk Memperbaiki Model

Jangan menebak-nebak tanpa arah saat modelmu gagal. Terapkan tindakan yang sesuai dengan diagnosis:

#### A. Jika Model Terkena Penyakit Bias Tinggi (*Underfitting*):
1. **Tingkatkan Kompleksitas Model**: beralihlah ke algoritma yang lebih fleksibel (misalnya ganti Regresi Linear dengan *Random Forest*, *XGBoost*, atau *Neural Networks*).
2. **Tambah Fitur Baru (*Feature Engineering*)**: beri model lebih banyak informasi relevan untuk dipertimbangkan.
3. **Turunkan Kekuatan Regularisasi**: perkecil nilai penalti penyusutan bobot (seperti memperkecil nilai parameter `alpha` di Ridge/Lasso atau mengurangi `dropout`).

#### B. Jika Model Terkena Penyakit Varians Tinggi (*Overfitting*):
1. **Kumpulkan Lebih Banyak Data**: perbanyak baris data latihan untuk menenggelamkan faktor kebetulan (*noise*).
2. **Terapkan Regularisasi**: pasang hukuman pada parameter bobot yang membesar (gunakan regularisasi L1/L2, hentikan latihan lebih awal lewat *Early Stopping*).
3. **Pangkas Jumlah Fitur (*Feature Selection*)**: buang kolom-kolom fitur yang tidak relevan atau redundan agar model tidak mempelajari korelasi palsu.
4. **Gunakan Metode Penggabungan (*Ensemble Bagging*)**: algoritma seperti *Random Forest* menggabungkan prediksi dari puluhan pohon keputusan yang berbeda untuk meredam goyahan varians secara signifikan.

---

### 4.3 Sumber Rujukan Topik 4
* **Zinkevich, Martin**, “Rules of Machine Learning: Best Practices for ML Engineering”, Google Research, Aturan #24: “Measure the delta between models”.
* **Géron, Aurélien (2022)**, *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*, Edisi ke-3, O'Reilly Media, Bab 11: “Training Deep Neural Nets (Regularization)”.

---

## 5. Mengapa Tarik-Ulur Ini Tidak Pernah Hilang?

Banyak pemula bertanya: 
> *“Apakah dengan kemajuan komputer super dan kecerdasan buatan masa kini, kita bisa menghilangkan tradeoff ini secara mutlak?”*

Jawabannya adalah **TIDAK**. Tarik-ulur bias dan varians adalah konsekuensi logis yang permanen dalam epistemologi sains data.

---

### 5.1 Teorema "No Free Lunch" (Wolpert & Macready, 1997)
Teorema terkenal dalam teori komputasi menyatakan bahwa: **tidak ada satu algoritma pun yang secara universal bekerja paling baik untuk segala macam permasalahan.**

* Setiap kali kita memilih sebuah model, kita harus menyuntikkan asumsi awal (**Bias Induktif**).
* Tanpa asumsi, model tidak bisa belajar menarik kesimpulan dari data masa lalu untuk data masa depan.
* Begitu kita memasukkan asumsi, kita otomatis membuka peluang terjadinya **Bias** jika asumsi kita meleset, atau memicu **Varians** jika kita membiarkan model terlalu bebas menerka-nerka tanpa asumsi.

### 5.2 Realitas Derau Dunia Nyata (*Irreducible Noise*)
Data dunia nyata tidak pernah bersih $100\%$. Sensor kamera memiliki bintik derau, alat ukur memiliki ketidakakuratan mikro, dan manusia yang mengetik label bisa melakukan kekeliruan. 

Karena derau (*noise*) selalu berbaur dengan sinyal pola asli, model yang berusaha sekuat tenaga menekan bias hingga nol pasti akan **tanpa sengaja mulai mempelajari derau tersebut**. Saat model menyerap derau, varians otomatis meledak naik.

Tugas praktisi Machine Learning bukanlah mencari model tanpa bias dan tanpa varians sama sekali, melainkan **menemukan titik keseimbangan terbaik (*sweet spot*)** yang meminimalkan total kesalahan pada skenario bisnis yang dihadapi.

---

### 5.3 Sumber Rujukan Topik 5
* **Wolpert, David H., dan Macready, William G. (1997)**, “No Free Lunch Theorems for Optimization”, *IEEE Transactions on Evolutionary Computation*, 1(1), hlm. 67–82.
* **Bishop, Christopher M. (2006)**, *Pattern Recognition and Machine Learning*, Springer, Bab 1: “Introduction (The Curse of Dimensionality and Model Selection)”.

---

## 6. Kerangka Kerja Diagnostik Cepat

Gunakan pohon alur di bawah ini untuk mengambil keputusan rekayasa saat mengevaluasi modelmu:

```
                            [ EVALUASI HASIL MODEL ]
                                       |
                                       v
                     +-----------------------------------+
                     | APAKAH ERROR DATA LATIH TINGGI?   |
                     +-----------------------------------+
                                    /     \
                             YA    /       \   TIDAK
                                  v         v
                      [ HIGH BIAS ]         +-------------------------------------+
                     (Underfitting)         | APAKAH ERROR DATA UJI JAUH LEBIH    |
                            |               | TINGGI DIBANDING DATA LATIH?        |
                            v               +-------------------------------------+
                   - Tambah fitur baru                      /     \
                   - Gunakan model lebih kompleks    YA    /       \   TIDAK
                   - Kurangi regularisasi                 v         v
                                                  [ HIGH VARIANCE ]  [ KONDISI IDEAL! ]
                                                   (Overfitting)     (Siap dirilis ke
                                                          |           peladen produksi)
                                                          v
                                                 - Tambah data latih
                                                 - Pasang regularisasi
                                                 - Pangkas fitur tak penting
```

---

## 7. Daftar Pustaka Terverifikasi

1. **Geman, Stuart, Bienenstock, Élie, dan Doursat, René (1992)**. “Neural Networks and the Bias/Variance Dilemma”. *Neural Computation*, 4(1), hlm. 1–58.
2. **Hastie, Trevor, Tibshirani, Robert, dan Friedman, Jerome (2009)**. *The Elements of Statistical Learning: Data Mining, Inference, and Prediction*, Edisi ke-2. Springer Science & Business Media.
3. **James, Gareth, Witten, Daniela, Hastie, Trevor, dan Tibshirani, Robert (2021)**. *An Introduction to Statistical Learning: with Applications in R*, Edisi ke-2. Springer.
4. **Goodfellow, Ian, Bengio, Yoshua, dan Courville, Aaron (2016)**. *Deep Learning*. MIT Press.
5. **Géron, Aurélien (2022)**. *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*, Edisi ke-3. O'Reilly Media.
6. **Bishop, Christopher M. (2006)**. *Pattern Recognition and Machine Learning*. Springer.
7. **Wolpert, David H., dan Macready, William G. (1997)**. “No Free Lunch Theorems for Optimization”. *IEEE Transactions on Evolutionary Computation*, 1(1), hlm. 67–82.
8. **Ng, Andrew (2018)**. *Machine Learning Yearning: Technical Strategy for AI Engineers in the Era of Deep Learning*. deeplearning.ai.
9. **Zinkevich, Martin**. “Rules of Machine Learning: Best Practices for ML Engineering”. Google Research.