# LIBRAS_CODE

Aplicacao para reconhecer datilologia de **Libras** (alfabeto A-Z) pela
webcam e mostrar a frase resultante em portugues na parte inferior da
janela. E a direcao Libras -> portugues; um avatar 2D (portugues -> Libras)
e trabalho futuro e nao faz parte do escopo atual.

## Por que esse desenho?

Nao ha dataset de Libras no projeto. Em vez de treinar um classificador que
sem dados teria acertarias arbitrarias, o reconhecimento e feito por
**regras geometricas**: quantos dedos estao estendidos e onde esta o polegar.

Varias letras tem a **mesma pose** e so se distinguem por orientacao da mao,
movimento ou trajetoria (A/I/G, J/Z, U/V, H/K, E/M/O/S/X, ...). O modulo de
datilologia devolve essas leituras como `candidates` com `needs` (o que falta)
em vez de chutar uma letra. O acumulador de frases ignora leituras ambiguas.

As regras tropecam nos pares de mesma pose (D/L, B/C): o desempate depende de
limiares medidos para uma mao e um angulo. Quando ha amostras rotuladas, um
**classificador** treinado nelas assume e generaliza melhor -- veja
[Treinar um classificador](#treinar-um-classificador). As regras continuam
como fallback e como unica via para as letras dinamicas (H, J, K, X, Y, Z).

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

## Treinar um classificador

O app funciona so com as regras. Para usar o classificador, grave amostras da
sua mao e treine:

```
python tools/collect_samples.py        # grava data/samples.npz
python tools/train_classifier.py       # treina models/dactylology_clf.npz
python app.py                          # carrega o modelo, se existir
```

Ao gravar, escolha a letra pelo teclado (A-Z) e mantenha a pose gravando; cada
letra quer ~20-40 amostras, variando um pouco angulo e distancia. O dataset
guarda os 21 landmarks (nao as features), entao sobrevive a mudancas no
extrator. O treino imprime acuracia e as confusoes mais comuns, segurando 20%
de cada letra para teste.

O modelo e uma regressao softmax em numpy puro (sem sklearn): o vetor de
entrada e o `HandFeatures.vector()`, padronizado por z-score. Em Libras 6
letras do alfabeto manual sao **dinamicas** (tem movimento): H, J, K, X, Y, Z.
Como o movimento nao aparece num quadro so, elas ficam de fora do modelo
estatico e seguem pelas regras com historico; o classificador cobre as outras
**20 letras estaticas**.

## Estrutura

```
libras/
  core/
    models.py          baixa o .task e extrai os dois .tflite
    hand_tracker.py    detector de palma + landmarker, em duas etapas
    features.py        features geometricas (estaticas) e temporais (movimento)
    phrase.py          acumulador de letras em frase (histerese temporal)
  signs/
    dactylology.py     regras A-Z com ambiguidades explicitas
    classifier.py      regressao softmax em numpy (opcional, com amostras)
  data/
    synthetic.py       gera poses de 21 landmarks sem camera
tools/
  collect_samples.py   grava amostras rotuladas (webcam) para treino
  train_classifier.py  treina o classificador a partir das amostras
  make_demo_video.py   gera um video sintetico "OLAR" para testar a app
app.py                 interface webcam
tests/                 unittest (tests/data/hands.jpg: foto real de referencia)
```

## Como a mao e encontrada

Em duas etapas, e a ordem importa. O `hand_landmarker.task` traz os dois
detectores, e o `landmarker` foi treinado com os recortes que o detector de
palma produz:

1. `hand_detector.tflite` (192x192) acha a **palma**: caixa e 5 keypoints
   (pulso e os quatro MCPs);
2. `hand_landmarks_detector.tflite` (224x224) roda sobre um recorte **girado**
   daquela palma, com o eixo pulso -> MCP do medio na vertical.

A diferenca medida e o que separava as duas explicacoes para o app nunca achar
a mao:

| recorte                                      | presenca |
|----------------------------------------------|----------|
| grade de busca livre (96-256 px, sem palma)  | 0.00-0.43 |
| ROI da palma, girada, 2.6x a caixa           | 0.89-0.99 |

Dois detalhes que custaram tempo e estao commented no codigo: o canvas de 192 e
um resize *esticado* do frame, entao num frame 640x960 a ROI e um retangulo e
nao um quadrado em pixels; e a grade de ancoras do detector de palma
(`48 x 24`, passos 4x8 px) teve de ser medida, nao vem documentada.

A `presence` do landmarker sozinha nao basta: uma ROI folgada demais devolve
presenca 0.9 com uma mao fantasma. Por isso `hand_tracker._quality` mede o
esqueleto em **pixels** (misturar escalas de x e y tornaria o limiar
dependente do aspecto do frame) e reprova relacoes impossiveis, como as quatro
falanges proximais apontando em direcoes opostas.

## Testes

```
python -m unittest discover -s tests
```

Cobrem: features invariantes a escala, dimensoes do vetor de features, as
regras de datilologia e suas ambiguidades, o acumulador de frases, o
classificador (treino, persistencia e integracao, com poses sinteticas), e o
rastreador contra uma foto real (`tests/data/hands.jpg`, duas maos) em
escala, espelhamento e proporcoes diferentes. Boneco desenhado nao serve como
fixture: o detector de palma nao enxerga mao sintetica.

## Estado e limitacoes

- Validado com video derivado de foto real, nao com webcam: o ambiente de
  desenvolvimento nao tem camera, entao `--camera` continua nao verificado.
- As 864 ancoras acima da primeira grade do detector de palma sao ignoradas
  (o mapeamento delas nao foi medido).
- O video de `tools/make_demo_video.py` serve para exercitar datilologia e
  frase, nao o rastreador.
- Sinais comuns (alem do alfabeto) e avatar 2D sao proximos passos.
