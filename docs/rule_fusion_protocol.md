# Kural tabanlı kanıt birleşimi deneyi

Bu protokol deneyden önce sabitlenmiştir. Amaç, bu yöntemden otomatik türetilen pas ve restart/çizgi kanıtlarını mevcut üç durumlu decoder'a düşük ağırlıkla bağlamayı incelemektir. Pas ve restart ayrı ve birlikte değerlendirilir. Ağırlıklar (0,5 ve 0,6) araştırma önkabulleridir; hedef maç etiketleriyle ayarlanmaz.

Geliştirme maçları: DFL J03WOH, J03WN1, J03WMX. Son kontrol: J03WR9, bu denemede yalnız aday ve seçim kuralı sabitlendikten sonra çalıştırılır. Bu maç önceki TCN araştırmalarında görülmüştür; tüm proje için hiç görülmemiş bir test değildir. DFL maçlarında Smart Tagging yoktur, hız konumlardan hesaplanır; farklı tracking ve etiket tanımları hedef maça aktarımı sınırlar.

Her maç temiz tracking ve hedef maçın yalnız top gözlemlenme maskesi aktarılmış tracking ile değerlendirilir. Gözlem maskesi hedef maçın top konumlarını ve possession etiketlerini içermez. Her varyantın iki koşul ve üç maç üzerindeki ortalama spellF1'i hesaplanır. Bir birleşim ancak ortalaması mevcut yöntemi geçerse ve maçların herhangi birinde iki koşul ortalaması mevcut yöntemden 0,02'den fazla kötü değilse aday olur; uygunsa en yüksek ortalamalı aday seçilir. Aksi halde mevcut yöntem seçilir. Diğer metrikler, özellikle kısa possession recall ve D F1, ayrı raporlanır; bunlar ek geliştirme hipotezleri üretebilir ama sabit son kontrol sonrasında ayar yapılmaz.

Public GT'ye erişim, final aday seçimi veya eşik ayarı yapılmaz. DFL referans etiketleri yalnız prediction tamamlandıktan sonra değerlendirmeye verilir. Program `scripts/evaluate_rule_fusion.py`; özel raporlar `data/runs/rule_fusion_protocol/` altında tutulur.
