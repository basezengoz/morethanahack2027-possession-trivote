# Dış veri ve lisans notu (15.11.2026)

## Metrica Sports sample-data

Resmi kaynaklar: [repo/README](https://github.com/metrica-sports/sample-data), [event tanımları PDF](https://github.com/metrica-sports/sample-data/blob/master/documentation/events-definitions.pdf), [Kloppy Metrica yükleyicisi](https://kloppy.pysport.org/user-guide/loading-data/metrica/).

Depoda 3 anonim örnek maç bulunuyor: Game 1–2 home/away tracking CSV'leri ve event CSV; Game 3 EPTS metadata XML + tracking text + JSON event. README, event/tracking senkronizasyonunu, 0–1 saha koordinatlarını ve 105×68 m boyutunu belirtiyor. Event CSV kolonları takım/tip/alt tip, periyot, başlangıç/bitiş frame ve saniye, oyuncu ve konum bilgilerini içeriyor. İndirme boyutları kaynak dosyalar alındığında yerel provenance kaydına yazılacak; burada tahmini boyut kullanılmıyor.

README sorumlu kullanım ve kamuya açık kullanımda Metrica'ya atıf koşulunu belirtiyor; ayrı, standart bir veri lisansı bulunamadı. Yeniden dağıtım ve türetilmiş model ağırlıklarının paylaşım koşulları bu metinde ayrıca açıklanmıyor. Bu aşama yerel araştırma/proxy değerlendirmesidir; veri ve türevler depoya eklenmez, bu kaynağa dayanan bir ağırlık final teslimine otomatik seçilmez. İlk kaynak incelemesinde veri indirilmedi; sonraki indirmeler yerel provenance kaydında izlenecek.

Events seyrek insan anotasyonlarıdır; frame başına possession ground truth değildir. PASS/RECOVERY/SHOT olayları yalnız dar, düşük ağırlıklı takım aday pencereleri üretebilir; SET PIECE yalnız restart bağlamı, BALL LOST/BALL OUT geçiş/oyun dışı sınır göstergesidir. Belirsiz pencereler maskelenmeli; tüm olay sürelerini possession veya D etiketine çevirmemeli.
