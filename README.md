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

| Tecla  | Acao                          |
|--------|-------------------------------|
| Q/ESC  | sai                           |
| Espaco | confirma a letra atual        |
| B      | apaga o ultimo caractere      |
| C      | limpa a frase                 |

Como e datilologia, uma letra precisa se **manter** na mesma pose por alguns
quadros (`--hold`, padrao 5) para entrar na frase, e depois de aceita ha um
`cooldown` para a mao poder sair e voltar. Uma pausa longa vira espaco entre
palavras.
