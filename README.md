# Scalable TalkingHead-Anime-V1

オリジナルのhttps://github.com/pkhungurn/talking-head-anime-3-demo をベースに

APIサーバ機能

任意サイズの切り出しとアップスケール機能

テンプレート自動作成機能

高度な抽象化されたポーズデータの指定

などの機能を追加し、サーバまたはクラスライブラリとして利用できるようにしました。
また、クライアント側では並列列処理機能を利用して処理時間を要するポースデータの生成とアップスケールをpipeline処理しています。更に自動瞬き機能やポーズゲネレーション機能も並列処理されてアプリからの操作を最小限にすることができます。

## リンク/クレジット

#### talking-head-anime-3-demo 

https://github.com/pkhungurn/talking-head-anime-3-demo

MIT License

#### anime-segmentation  背景削除

https://github.com/SkyTNT/anime-segmentation

Apache License  Version 2.0,

#### Real-ESRGAN  アップスケーラ

https://github.com/xinntao/Real-ESRGAN

BSD 3-Clause License

#### Anime-Face-Detection

https://github.com/animede/anime_face_detection

MIT License


## Scalable-Tanking-Head-Anime-3-API との違い

Scalable-Tanking-Head-Anime-3-APIはアップスケーラやAnimeFaceDetection,背景削除を独立したサーバで動かしています。とても面倒でした。またテストはテストプログラムで動きを確認するだけでした。Scalable TalkingHead-Anime-V1で独立して動いていた各AIサーバをメインサーバ機能のsubprocessとして実行・停止を行いため、サーバ起動は一つのPythonファイルの実行でずべて動くように大きく改良してイます。またテスト用のGUIも準備しました。Scalable-Tanking-Head-Anime-3-API で使えたテストプログラムはすべて動きます。またクラスライブラリに変更は無いので過去のバージョンと互換です。GUIコードではクラスライブラリの使い方がわかるように簡略化を避けました。

## 詳細な使い方について

使い方が多岐に渡るため、めぐチャンネルにて順次記事にします。技術書典でご購入いただいた第2部の補完及び機能アップの説明になります。

## 確認済み動作環境
Ubuntu22.04 / 20.04

CPUはそれなりに処理能力を要します。マルチプロセッシングや複数のFastAPIサーバを動かすため、コア数の多いi5以上を推奨します。

GPUは必須です。　x4以上のアップスケールを行うためにはRTX3060以上が必要です。x1で使う場合は1650程度でも動きます。VRAM容量は2G以上ですが、扱うキャラクタ画像（最大20）が増えるとVRAMを使うため概ね4G程度を目処にしてください。


## インストール

インストール.txtに記載しています。特にCUDAのインストールについては記述ありませんが、12.3で動作確認をしています。仮想環境を作成するので事前にvenv環境が作成出来る準備が必要です。

### CUDAの確認
nvcc -V　　このコマンドにて以下のようなバージョン情報が表示されればOKです。

nvcc: NVIDIA (R) Cuda compiler driver

Copyright (c) 2005-2023 NVIDIA Corporation

Built on Wed_Nov_22_10:17:15_PST_2023

Cuda compilation tools, release 12.3, V12.3.107

Build cuda_12.3.r12.3/compiler.33567101_0


### リポジトリのクローン

git clone git@github.com:animede/Scalable-Tanking-Head-Anime-3-API.git

### 仮想環境の作成とインストール
python3 -m venv tkh

source tkh/bin/activate

cd Scalable-talking-head-anime-v1

pip install requirements.txt

### ウエイトのダウンロード
#### Talking-Head-Anime3

wget https://www.dropbox.com/s/y7b8jl4n2euv8xe/talking-head-anime-3-models.zip?dl=0

または
HuggingFace
https://huggingface.co/UZUKI/Scalable-tkh
からダウンロード

dataフォルダにtalking-head-anime-3-models.zip?dl=0をコピーし、そこで展開
作成されたホルダー名をmodelsに変更

#### その他もHuggingFaceからダウンロード
ssd_best8.pthをweightsホルダへコピー

realesr-animevideov3.pthをweightsホルダへコピー

isnetis.ckptはtalking-head-anime-3-v1-apiのTOPへコピー

### Webアプリの動かし方
thh_全機能の動かし方に記載しています。

#### Scalable TalkingHeadAnime-V1サーバの起動
ターミナルを開いて以下のコマンドを実行

source tkh/bin/activate

cd Scalable-talking-head-anime-v1

python poser_api_v1_3S_server.py

#### Webサーバの起動
ターミナルを開いて以下のコマンドを実行

source tkh/bin/activate

cd Scalable-talking-head-anime-v1

python tkh_gui_html.py

host="0.0.0.0", port=3001で起動するので、ブラウザーから127.0.0.1:3001にアクセスします。

ネットワーク上の他のPCからもアクセスできますいが、複数クライアントへの正常なサービスができません。

URLとポートはtkh_gui_html.pyの最後の行で変更可能です。

#### 基本的な使い方
ブラウザでアクセスするとファイル選択ボックスが表示されるのでクリックして使いたい画像をファイルを選択してください。選択後は自動的にテンプレートまで作成します。キャラクタが任意の位置に一人だけいる使える画像であればほとんど利用できます。背景があっても、背景が透過であっても問題ありません。


テンプレート画像が表示されたら、Generate Imageボタンが表示されます。クリックするとmodeに従った生成画像が表示されます。黒い画面になる場合はもう一度Generate Imageボタンをクリックしてください。生成画像が表示されるとともに、下に操作パネルが表示されます。

キャラクタを変えるには再度ファイル選択からやり直してください。


