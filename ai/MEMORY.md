# MELANOMA PROJECT — Session Memory Index

> **Konvansiyonlar:** Her seans başında bu dosyayı oku, ilgili detay dosyasına git.
> "Devam et" dersen buradaki bağlamla devam edilir.

## Dosyalar

- [project_overview.md](project_overview.md) — Proje özeti, yöntemler, sonuçlar, GÜNCEL DURUM
- [methodology_flaws.md](methodology_flaws.md) — Tespit edilen tüm metodolojik hatalar (öncelik sırasıyla)
- [fix.md](fix.md) — Detaylı düzeltme planı + her fix'in durumu ✅/❌
- [conference_info.md](conference_info.md) — TOK 2026 konferans bilgileri ve takvim
- [user_profile.md](user_profile.md) — Zeynep hakkında notlar

## Hızlı Durum

| Konu | Durum |
|---|---|
| Proje | HAM10000 melanoma, binary, 10 yöntem kıyaslaması |
| En iyi sonuç (eski) | Ensemble F1=0.6967, AUC=0.954 |
| Hedef | F1≥0.78 (ISIC 2019 + aug ile ~0.78-0.84 bekleniyor) |
| Konferans | TOK 2026 — Ankara Üniversitesi — 8-10 Ekim 2026 |
| Bildiri son tarihi | **15 Ağustos 2026** |
| GitHub | public repo, tüm değişiklikler push edildi |

## Teknik Ortam (Kritik)

- **Eğitim:** Google Colab Pro+ (A100 40 GB)
- **GitHub bağlı:** Evet — Colab'da her seans `!git pull` yap
- **Drive path:** `MyDrive/melanoma/` — tüm data, checkpoint, result buraya yazıyor
- **X_all.npy:** 224×224 kaydedilmiş (config.IMG_SIZE=448 ama GERÇEK boyut 224)
- **X_isic2019_mel.npy:** Drive'da var (~2.7 GB, 224×224'e resize edilmiş)
- **Recovery:** Her CNN notebook sonunda recovery cell var
- **Seed:** 42 — asla değiştirme

## Seans Notu (2026-06-20)

Bu seansta yapılanlar:
- FIX 1: ISIC 2019 entegrasyonu tamamlandı (12_isic2019_prep.ipynb + load_arrays_extended)
- FIX 2: Augmentation güçlendirildi (config.py + data.py + notebooks/04-09)
- FIX 3,7,8,9: 10_aggregation.ipynb güncel (threshold, CI, ablation, literatür)
- FIX 4: VGG diagnostic hücresi (05_vgg16_bn.ipynb)
- FIX 5: XGBoost diagnostic (11_hybrid_fusion.ipynb)

**Sıradaki görev:** CNN'leri Colab A100'de yeniden eğit (04→09), sonra 10 ve 11'i çalıştır.

## Önemli Sayılar

```
Total images:   10,015  (7 class → binary)
Training:        7,008  (mel: ~779 HAM + ~4,522 ISIC 2019 = ~5,301 total, non-mel: ~6,229)
Val:             1,503  (mel: ~167, saf HAM10000)
Test:            1,504  (mel: 167, non-mel: 1,337, saf HAM10000)
Best F1 (eski):  0.6967 (ensemble)
Best AUC (eski): 0.9540 (ensemble)
Hedef F1:        0.78 ~ 0.84
```
