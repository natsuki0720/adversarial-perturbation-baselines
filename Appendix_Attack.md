# 本リポジトリで使用した敵対生成手法の理論解説
本リポジトリでは，以下の3種類の敵対的摂動生成手法を実装・比較した．

## FGSM（Fast Gradient Sign Method）
- **目的**：単一の入力画像 $x$ に対して，正解ラベル$y$の分類精度を下げる．
- **定式化** :
$$
x^{\text{adv}} = x + \varepsilon \cdot \text{sign}(\nabla_x \mathcal{L}(f_\theta(x), y)) 
$$
- **特徴と式の直感的解釈**:
    - 入力に対する損失関数の勾配方向に摂動を加える 
    - わずかな摂動でモデルを効率よく騙すことができる
    - **高速・簡易**だが，入力ごとに再生成が必要
    - モデルの構造・勾配情報に強く依存

### FGSMに関する解説・考察・検討
#### 攻撃の効率性について
FGSMはモデルの入力に対する**損失関数の勾配方向に摂動を加える攻撃手法**である．  
モデルが最も敏感な勾配方向に摂動を加えるため，最も効率的に分類境界を超える摂動を与えることが可能となるため，わずかな摂動でモデルに誤分類させることが可能となる．これが，**FGSMの効率的な攻撃性能の源泉**となる．

#### 転移性・実用性について
一方で，この方法の課題としては**モデルごとの勾配情報に強く依存する**方法であるため，実運用モデルと攻撃者が持つモデルの勾配が大きく異なるとその転移性も制限を受ける点が課題となる.  
また，FGSMは各入力に対して**個別に敵対的な摂動を加える手法**であることにも留意する必要がある．現実の問題を考える時に，入力対象$x$がすでに判明していてかつ，攻撃者が事前に入力にアクセスできる状態は考えにくいため，**攻撃手法としての活用することは現実には困難**だと考えられる．  

#### FGSMの活用方法に関する検討
さて，FGSMを有効活用する手段としてはどのようなものがあるか検討する．  
最も有効的に活用する手段としては，データ拡張としての活用が考えられる．一般にデータ拡張は，訓練データが不足するときの補完手段として行われるが，FGSMの持つ「不安定な方向を炙り出す」性質を活用し，**モデルの頑健性向上**を図る用途での使用を考える．    
FGSMは損失関数の勾配方向に摂動を加えることで，効率よく敵対的な摂動を加える攻撃手段であった．
> FGSMで得られる摂動方向は，モデルが**推論不安定性を持つ方向**を明示すると考えられる．

そこで，このような方向へ摂動を加えたのサンプルを訓練データに加えることで**損失関数の勾配を平滑化**し，推論安定性を向上させる可能性が期待できる．
これにより，入力の変化に対するsoftmax出力の変動が小さくなるため，モデルの推論の安定性が高まると考えられる．  
具体的には，FGSM摂動付きサンプル$x^{\text{adv}}$を訓練データに追加し再学習を行う．これにより，モデルのsoftmax出力は摂動に対して鋭い変化をしなくなり，ロバスト性の向上が期待できる．

※注意点  
以上のデータ拡張には**類似データの重複学習**が生じ，特定のデータに対する過学習を助長するリスクが考えられる．  
対策としては
- 摂動付与データと元データの差し替え
- 摂動データによる低学習率でのチューニング
- FGSMを適用する前のデータを訓練データから分離する
などが考えられる．

## UAP（Universal Adversarial Perturbation）
- **目的**： 全ての入力（$x \sim D$）に共通して適用可能な**汎用摂動**$\delta$を学習し，**多様な入力に対して誤分類を誘発すること**を目指す．

- **定式化**:
  $$
  \delta^* = \argmax_{\delta \in \mathcal{C}_\epsilon} \; \mathbb{E}_{(x, y) \sim D}[\mathcal{L}(f_\theta(x + \delta), y)]
  $$
  - $\mathcal{C}_\epsilon$：摂動の大きさを制限する集合

- **特徴**
    - 訓練データ全体に共通して通用する一つの摂動テンソルを最適化
    - 摂動テンソルは訓練データに対する損失関数を制約下で最大化することを目指す
    - 制約を厳しくすることで，視覚的な判断が困難な摂動テンソルが生成可能に
    - 一度学習すれば，新しい入力にも適用可能．
    - モデルの決定境界の歪みに共通して作用する摂動を最適化する構造

### UAPに関する解説・考察・検討
#### 汎用摂動の意味と直感
UAPは，ある一つの摂動テンソル$\delta$を通じて，**モデルそのものの脆弱性**を突くことを目的とした攻撃手法である．UAPは入力に依存せず，モデルが分類を誤るように全ての入力に同様の摂動を加えることで，汎用的な摂動テンソルを生成する．

#### 転移可能性について
本リポジトリにおいて，UAPは高い転移可能性を示した．特に`ResNet`に対する転移性能には目を見張るものがあり，小規模データから十分有効な敵対生成を行える可能性が示唆される．特定のモデルの損失の最大化を図っただけであるにも関わらず，構造が異なり，より豊富な訓練データを用いたモデルに対して有効性を示した点は，敵対事例の転移性に関しての考察の重要性を与えるとともに，セキュリティ上の重要な問題になりうる．  

UAPが他のモデルにも転移する背景としては，訓練済みモデルが入力空間に形成する決定境界が，モデル構造が異なっても類似した幾何構造を持つためであると考えられる．特に今回使用したCIFAR-10などの比較的単純なデータセットにおいては，複数のモデルが類似した分類境界を学習する傾向にあるため，UAPの転移性が観測されるのは不思議ではない．

#### 実用性について
UAPの最も大きな危険性は，**事前に生成した1つの摂動テンソルが広範な入力に有効**である点にある．   
また現実においては，入力に依存せず一貫した摂動を物理的に与える手法がUAPのような攻撃と極めて相性がよく，リアルタイム認識や異常検知といったタスクにおけるセキュリティリスクが無視できない．  
物理空間でのUAPの応用事例としては敵対摂動に対応するようなフィルムなどをカメラに直接貼り付けることで，異常検知システムなどへの妨害が考えられる．本リポジトリで示したような，人間が視覚的に気づけないような攻撃も可能であると考えられるため，物理空間からのAIに対する敵対行動に対する対策の検討も重要になると考えられる．  


## Local UAP
- **目的**：画像のごく一部の領域（例：左上8x8ピクセルなど）に限定して摂動を施すことで，**モデル全体の判断を誤らせる汎用的な攻撃テンソル**を学習する
- **定式化**：
  $$
  \delta^* = \argmax_{\delta \in \mathcal{C}_{\varepsilon}} \; \mathbb{E}_{(x,y) \sim D}[\mathcal{L}(f_{\theta}(x + M \odot \delta),y)] 
  $$  
  - $\delta \in \mathbb{R}^{C \times H \times W}$：摂動テンソル
  - $M$：局所領域のみ1，それ以外が0のマスクテンソル
  - $\odot$：要素積

- **特徴**:
  - 摂動を画像全体ではなく，**限定された一部分**のみに加える
  - 実空間での「ステッカー攻撃」などを模倣できる
  - 本リポジトリでは，UAPと異なり，視覚的に摂動が目指すことを許容した
  - 攻撃箇所を制御できるため，**物理空間での実装に近い応用が可能**
  - 局所性ゆえ，転移性はやや低い傾向にある

- **損失関数について**
敵対生成全般に関連する話題だが，敵対的摂動は損失関数を**最大化**することで生成されるため，対称とする損失関数の設計は極めて重要となる．本リポジトリではLocal UAPの訓練に以下の3つの損失関数を使用した．なお，これらは通常の深層学習においては最小化されるが，敵対的摂動の生成においては**入力$x$加える摂動$\delta$を，各損失関数が最大化される方向に求める**．  
  - Cross Entoropy
    $$
      \mathcal{L}_{CE} = \mathbb{E}_{(x,y) \sim D}[-log(p_y^{(x + \delta)})]
    $$
    - クラス分類タスクにおける標準的な損失関数
    - この関数の増加は，正解ラベルの予測確率の低下に対応する
  
  - Confidence Penalty
    $$
      \mathcal{L}_{\text{confPenalty}} = \mathcal{L}_{\text{CE}} - \lambda_{\text{conf}} \cdot \mathbb{E}_{(x,y) \sim D}[p_y(x + \delta)]
    $$
    - クロスエントロピー誤差に，分類結果に対する信頼度を付与したもの
    - この関数の最大化では，正解ラベルの予測確率の低下に加え，Softmax出力が不確かになるような摂動を獲得する
  
  - Entropy Penalty
      $$
      \mathcal{L}_{\text{entPenalty}} = \mathcal{L}_{\text{CE}} + \lambda_{\text{ent}} \cdot \mathbb{E}_{x \sim D}\left[\sum_{c=1}^{C} p_c(x + \delta) \log p_c(x + \delta) \right]
      $$
    - クロスエントロピー誤差に加え，エントロピーを追加したもの
    - この関数の最大化は，単なる正解ラベルの予測確率の低下を超え，モデルが何も予測しない（出力確率分布の平滑化）を促す摂動の生成に対応する

### Local UAPに関する解説・考察・検討
#### 局所摂動の意味と直観
Local UAPはUAP同様に汎用的な摂動の生成を試みるアプローチである．本リポジトリでは，入力画像$x$の左上8x8ピクセルの領域のみに限定して，UAPによる攻撃を試みた．局所的な領域に限定した攻撃でモデルに誤分類を促すことを目的としている．  
対象物が中心にある画像では，一部領域が欠損していても人間には何が映った画像かの判断に永享がないのに対し，分類器は誤った選択をするケースがあることから，モデルの画像の解釈と人間の視覚的解釈には根本的な違いがあることが示唆される．  
現実的な活用事例としては，画像認識の対象物に対するステッカーの貼り付けによるモデルの誤認識の誘発などが考えられる．

#### 転移可能性について
本リポジトリでの検討では，`ResNet18`に対して，`CNN_small`で生成したLocal UAPの転移性が低かった．この要因としては，`ResNet18`が残差接続構造により大域情報を保持する機構を有していたため，局所領域への攻撃の影響を充分に吸収できた可能性が考えられる．  
CNNは局所性のある情報が徐々に伝播していく構造となるため，局所性のある攻撃が有効に働いた一方で，残差接続的に元の入力情報を保持される機構に対しては相性が悪かったと考えられる．  
この結果はViT（Vision Transformer）などに対する局所的な入力の破壊に対するロバスト性を示唆する結果となる．


