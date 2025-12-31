# Agency Move Model - Developer Notes

## sample

```sh
python -m gtaz.screens -i -k -m -f 10 -d 0
```

Auto save samples at:
- `src/gtaz/cache/actions/`

Select goode samples, and copy to:
- `src/gtaz/cache/agency_move/`
- this step requires manual work

## train

```sh
python -m gtaz.agency_move.train -m train -e 30 -b 64 -w
```

Auto save latest and best model at:
- `src/gtaz/checkpoints/agency_move/`

## export and inference

```sh
python -m gtaz.agency_move.inference -m export -w
```

Auto save .onnx and .engine at:
- `src/gtaz/checkpoints/agency_move/`

## realtime run

```sh
python -m gtaz.agency_move.realtime
```