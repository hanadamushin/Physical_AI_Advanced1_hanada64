"""Build a self-contained Colab notebook and a small shareable source package."""
from pathlib import Path
import json, zipfile

here=Path(__file__).resolve().parent
root=here.parent
cells=[]
def cell(kind,text):
    value={'cell_type':kind,'metadata':{},'source':text.splitlines(True)}
    if kind=='code':value.update(execution_count=None,outputs=[])
    cells.append(value)

cell('markdown','''# EEGジェル処置アシスト：頭の周囲を移動するデモ

10–20法の標準位置から選んだ16電極版です。CPUランタイムで「すべてのセルを実行」してください。GPUやAPIキーは不要です。
配置：Fp1/Fp2, F7/F3/F4/F8, T7/C3/C4/T8, P7/P3/P4/P8, O1/O2。
模型の鼻根点・後頭点・左右耳前点を基準に、楕円体上の弧長を10%/20%に分割して配置します。個人の頭部実測値ではありません。左右対称の理想化した模型です。32電極への拡張では追加位置を模型上で補間します。

**今回の目標**：頭の周囲を移動し、模擬インピーダンスに応じて処置・再測定を繰り返す。

頭を大きく表示し、首から下は省略しています。台座は顔の正面90°を空けた後方270°の弧状レール上を往復します。額の電極には弧の端からアームの向きを変えて届くよう、肩と手首に旋回軸を設けています。270°制限は台座の移動範囲です。
注射器本体はアームに固定し、空の状態から補給容器へ移動して吸引します。残量不足なら再補給します。ピストンには実際に動くスライド関節を設け、吸引では引き、注入では押します。ピストン位置・注射器内のジェル・残量が連動し、同じ3Dモデルの拡大映像も表示します。
MuJoCo上の運動学デモで、実際のインピーダンス計測やモータ・力制御は行いません。
ジェルの作用は仮定した数式です。実測の改善や人体への安全性を示すものではありません。
表示の10 kOhm、処置量mLはデモの任意の設定です。

コードと動画生成はWindows上で確認済みです。また、作成者本人からColabですべてのセルを最後まで実行できたとの報告を受けています。
''')
cell('code','''# 1. 描画に必要なライブラリを準備（CPUによる画面なし描画）
import os, sys, subprocess
subprocess.run(['apt-get','-qq','update'],check=True)
subprocess.run(['apt-get','-qq','install','-y','libosmesa6'],check=True)
subprocess.run([sys.executable,'-m','pip','install','-q','mujoco==3.17.0','imageio==2.38.1','imageio-ffmpeg==0.7.0','numpy>=1.26','Pillow>=10'],check=True)
os.environ['MUJOCO_GL']='osmesa'
os.environ['PYOPENGL_PLATFORM']='osmesa'
print('準備完了')
''')
cell('markdown','''## AIとフィードバック

合成データ1,600例でリッジ回帰モデルを学習し、現在値と処置量から改善量を予測します。別の合成データ500例で予測誤差を確認します。候補量から目標に届く最小量を選択し、処置後の観測で電極ごとの補正係数を更新します。

巡回順序・閾値・停止条件はルールです。16電極のうち初めから目標内の4電極は処置しません。T7は意図的に改善しない設定で、2回処置後に要確認として停止します。
10–20法の基準：https://pubmed.ncbi.nlm.nih.gov/28436791/

モデルの学習・評価・実行環境は同種の合成仮定に基づき、実データでの性能評価ではありません。
''')
cell('code','%%writefile demo.py\n'+(here/'demo.py').read_text(encoding='utf-8'))
cell('code','''# 2. 経路・離隔確認と動画生成
SEED=42
FPS=12
ELECTRODES=16 # 次の拡張では32に変更可能（動画尺は長くなります）
PLAYBACK=2   # 2倍速。動画内にも明記
subprocess.run([sys.executable,'demo.py','--seed',str(SEED),'--fps',str(FPS),'--electrodes',str(ELECTRODES),'--playback',str(PLAYBACK),'--out','output'],check=True)
''')
cell('code','''# 3. 動画と処置結果を確認
from IPython.display import Video, display
import json
from pathlib import Path
display(Video('output/eeg_gel_demo.mp4',embed=True,width=900))
summary=json.loads(Path('output/summary.json').read_text())
print('目標内:',summary['passed'],'/',summary['electrodes'])
print('要確認:',summary['review'])
print('処置回数:',summary['visits'])
print('台座の移動範囲:',summary['base_arc_degrees'],'度')
print('補給回数:',summary['refill_count'])
print('処置ログ: output/events.csv')
''')
cell('markdown','''## 結果の読み方

- 赤：模擬インピーダンスが目標超過、緑：目標内、橙：改善せず要確認。
- 動画左上：前が上、被計測者の左が左になる電極マップ。3Dの正面像では被計測者の左が画面右に見えます。動画内に見える電極名と頭皮ガイドを追加しています。水色の小さな塊は抽象化したジェルです。
- 各描画フレームで関節可動域とロボットから模型までの離隔を確認します。自己衝突や連続時間の衝突回避、人体の安全性は検証していません。
- 16電極の既定条件では2倍速で約177秒、20回処置、補給7回（初回を含む）、最終15電極が目標内、T7は要確認です。画面右上はシミュレーション時刻を表示します。
- 補給容器は模型の後方です。処置先・補給・終了後の駐機を通して、台座は後方270°の範囲内を移動します。
- 右上は模型内の注射器を別のカメラで拡大した映像です。PULLは吸引、PUSHは注入、HELDはピストンを保持している状態です。表示の残量とスライド関節・筒内のジェルが連動します。ストロークと外形は見やすさのための仮寸法で、実際の容量校正や圧力・流体の検証ではありません。

課題提出では、コードを公開GitHub、動画をYouTubeへ掲載し、PDFレポートとURLをOmniCampusへ提出します。ノートブック自体はColabのファイルメニューから保存してください。
''')
cell('code','''# 4. 動画・コード・ログをまとめて保存
import zipfile
with zipfile.ZipFile('EEG_Gel_Assist_results.zip','w',zipfile.ZIP_DEFLATED) as archive:
    archive.write('demo.py')
    for path in Path('output').glob('*'):
        if path.is_file(): archive.write(path)
from google.colab import files
files.download('EEG_Gel_Assist_results.zip')
''')
nb={'cells':cells,'metadata':{'colab':{'name':'EEG_Gel_Assist_Colab.ipynb'},'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5}
for i,c in enumerate(cells):c['id']=f'eeg-gel-{i:02d}'
notebook=here/'EEG_Gel_Assist_Colab.ipynb'
notebook.write_text(json.dumps(nb,ensure_ascii=False,indent=2),encoding='utf-8')
for c in cells:
    if c['cell_type']=='code':
        source=''.join(c['source'])
        if source.startswith('%%writefile'):source=source.split('\n',1)[1]
        compile(source,'notebook-cell','exec')
with zipfile.ZipFile(root/'EEG_Gel_Assist_v07_piston.zip','w',zipfile.ZIP_DEFLATED) as archive:
    for p in here.iterdir():
        if p.is_file() and p.name!='build_package.py':archive.write(p,'eeg_gel_demo/'+p.name)
    for p in ((root/'output_v07').iterdir() if (root/'output_v07').exists() else []):
        if p.is_file() and p.suffix in ['.mp4','.png','.json','.csv','.xml','.npz']:
            archive.write(p,'output/'+p.name)
    if (root/'verification_v07/checks.json').exists():
        archive.write(root/'verification_v07/checks.json','verification/checks.json')
print('Notebook syntax checked; source/video ZIP created.')
