Bundled win-probability models (`de_<map>.json`).

The Mirage model trained on 7,393 ESTA rounds was lost when the cloud VM recycled
(no git remote). Re-train with:

```
poswing esta esta/data/lan -o cache/esta --map de_mirage --workers 3
poswing fit cache/esta -o src/poswing/assets/models/de_mirage.json --map de_mirage
```

Until then the UI fits a model on the uploaded demo itself.
