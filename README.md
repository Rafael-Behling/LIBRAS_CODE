# LIBRAS_CODE

Aplicacao para reconhecer datilologia de **Libras** (alfabeto A-Z) pela
webcam e mostrar a frase resultante em portugues na parte inferior da
janela. E a direcao Libras -> portugues; um avatar 2D (portugues -> Libras)
e trabalho futuro e nao faz parte do escopo atual.

## Instalacao

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Modelos (MediaPipe hand landmarker) sao baixados automaticamente em `models/`
na primeira execucao. O `.task` e um zip: o app extrai os dois detectores
(`hand_detector.tflite` e `hand_landmarks_detector.tflite`), e os dois sao
necessarios.

## Uso

```
python app.py --camera 0       # webcam 0
python app.py --source video.mp4
python app.py --source video.mp4 --no-gui   # sem janela, so no terminal
```

Controles na janela de video:

| Tecla   | Acao                                     |
|---------|-------------------------------------------|
| Q/ESC   | sai                                      |
| Espaco  | confirma a letra atual                    |
| B       | apaga o ultimo caractere                  |
| C       | limpa a frase                             |
| ENTER   | salva a frase no historico (H abre)      |
| + / -   | sobe / desce o piso de confianca          |
| [ / ]   | encolhe / amplia a margem de votos        |
| T       | alterna o tema claro/escuro               |
| ?  /    | abre a lista de atalhos                   |

Os ajustes de `+ / - / [ / ]` ficam salvos em `models/tuning.json` e valem
nas proximas execucoes. Com `--log-misses data/misses.npz`, apagar com `B`
uma letra que acabou de entrar grava os landmarks dela no arquivo; depois

```
python tools/relabel_misses.py data/misses.npz --as <letra certa>
```

rotula os registros e anexa ao dataset de treino (`data/samples.npz`) --
o ciclo de melhora continua com `tools/train_classifier.py`.

Como e datilologia, uma letra precisa se **manter** na mesma pose por alguns
quadros (`--hold`, padrao 5) para entrar na frase, e depois de aceita ha um
`cooldown` para a mao poder sair e voltar. Uma pausa longa vira espaco entre
palavras.

## Modelos

O classificador estatico (`models/dactylology_clf.npz`) cobre as 20 letras
estaticas; e treinado com

```
python tools/train_classifier.py
```

As 6 letras dinamicas (H, J, K, X, Y, Z) tem movimento e nao cabem num
quadro so. O modelo temporal (`models/dactylology_motion_clf.npz`) cobre
elas, e precisa de sequencias no dataset (cada rajada de gravacao do
`tools/collect_samples.py` vira um grupo):

```
python tools/train_classifier.py --motion
```

O detector de palma roda a cada `--palm-every` quadros (padrao 2) enquanto
uma mao esta seguida, e a cada quadro quando nao ha mao -- o landmarker
roda sempre.
