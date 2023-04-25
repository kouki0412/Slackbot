from tokenize import Double
from slackbot.bot import respond_to  #@メッセージへの応答
from slackbot.bot import listen_to      #チャンネル内発言への応答
from slackbot.bot import default_reply    # デフォルトの応答
import slackbot_settings

###################################################
#この下にSVM学習済みファイルと単語リスト読み込み部分を貼り付ける             

import pickle

# 保存したモデルをロードする
filename = "svmclassifier.pkl"
loaded_classifier = pickle.load(open(filename, "rb"))

# 単語リストを読み込みリストに保存
basicFormList = []
bffile = "basicFormList.txt"
for line in open(bffile, "r", encoding="utf_8"):
    basicFormList.append(line.strip())
print(len(basicFormList))

###################################################
#この下にクラスや関数を貼りつける

from janome.tokenizer import Tokenizer

# 単語のクラス
class Word:
    def __init__(self, token):
        # 表層形
        self.text = token.surface

        # 原型
        self.basicForm = token.base_form

        # 品詞
        self.pos = token.part_of_speech
        
    # 単語の情報を「表層系\t原型\t品詞」で返す
    def wordInfo(self):
        return self.text + "\t" + self.basicForm + "\t" + self.pos

# 引数のtextをJanomeで解析して単語リストを返す関数
def janomeAnalyzer(text):
    # 形態素解析
    t = Tokenizer()
    tokens = t.tokenize(text) 

    # 解析結果を1行ずつ取得してリストに追加
    wordlist = []
    for token in tokens:
        word = Word(token)
        wordlist.append(word)
    return wordlist

import random

# キーワード照合ルールのリスト（keywordMatchingRuleオブジェクトのリスト）
kRuleList = []

# 応答候補のリスト（ResponseCandidateオブジェクトのリスト）
candidateList = []

# キーワード照合ルールのクラス（キーワードと応答の組み合わせ）
class KeywordMatchingRule:
    def __init__(self, keyword, response):
        self.keyword = keyword
        self.response = response

# 応答候補のクラス（応答候補とスコアの組み合わせ）
class ResponseCandidate:
    def __init__(self, response, score):
        self.response = response
        self.score = score
    def print(self):
        print("候補文 [%s, %.5f]" % (self.response, self.score))

# キーワード照合ルールを初期化する関数
def setupKeywordMatchingRule():
    kRuleList.clear()
    for line in open('kw_matching_rule.txt', 'r', encoding="utf_8"):
        arr = line.split(",")    
        # keywordMatchingRuleオブジェクトを作成してkRuleListに追加
        kRuleList.append(KeywordMatchingRule(arr[0], arr[1].strip()))
        
# キーワード照合ルールを利用した応答候補を生成する関数
def generateResponseByRule(inputText):
    for rule in kRuleList:
        # ルールのキーワードが入力テキストに含まれていたら
        if(rule.keyword in inputText):
            # キーワードに対応する応答文とスコアでResponseCandidateオブジェクトを作成してcandidateListに追加
            cdd = ResponseCandidate(rule.response, 1.0 + random.random())
            candidateList.append(cdd)

# ユーザ入力文に含まれる名詞を利用した応答候補を生成する関数
def generateResponseByInputTopic(inputWordList):
    # 名詞につなげる語句のリスト
    textList = ["は好きですか？", "って何ですか？"]
    
    for w in inputWordList:
        pos2 = w.pos.split(",")
        # 品詞が名詞だったら
        if pos2[0]=='名詞':
            cdd = ResponseCandidate(w.basicForm + random.choice(textList), 
                                    0.7 + random.random())
            candidateList.append(cdd)
            
# 無難な応答を返す関数
def generateOtherResponse():
    # 無難な応答のリスト
    bunanList = ["なるほど", "それで？"]

    # ランダムにどれかをcandidateListに追加
    cdd = ResponseCandidate(random.choice(bunanList), 0.5 + random.random())
    candidateList.append(cdd)
    
from collections import Counter

# 単語情報リストを渡すとカウンターを返す関数
def makeCounter(wordList):
    basicFormList = []
    for word in wordList:
        basicFormList.append(word.basicForm)
    # 単語の原型のカウンターを作成
    counter = Counter(basicFormList)
    return counter

# Counterのリストと単語リストからベクトルのリストを作成する関数
def makeVectorList(counterList, basicFormList):
    vectorList = []
    for counter in counterList:
        vector = []
        for word in basicFormList:
            vector.append(counter[word])
        vectorList.append(vector)
    return vectorList  

from sklearn import svm

# ネガポジ判定の結果を返す関数
# 引数 text:入力文, classifier：学習済みモデル, basicFormList：ベクトル化に使用する単語リスト
def negaposiAnalyzer(text, classifier, basicFormList):
    # 形態素解析して頻度のCounterを作成
    counterList = []
    wordlist = janomeAnalyzer(text)
    counter = makeCounter(wordlist)
    
    # 1文のcounterだが，counterListに追加
    counterList.append(counter)

    # Counterリストと単語リストからベクトルのリストを作成
    vectorList = makeVectorList(counterList, basicFormList)

    # ベクトルのリストに対してネガポジ判定
    predict_label = classifier.predict(vectorList)

    # 入力文のベクトル化に使用された単語を出力
    for vector in vectorList:
        wl=[]
        for i, num in enumerate(vector):
            if(num==1):
                wl.append(basicFormList[i])
        print(wl)

    # 予測結果を出力
    print(predict_label)

    # 予測結果によって出力を決定
    if predict_label[0]=="1":
        output = "よかったね"
    else:
        output = "ざんねん"

    return output

def generateNegaposiResponse(inputText):
    # ネガポジ判定を実行
    output = negaposiAnalyzer(inputText, loaded_classifier, 
                              basicFormList)
    
    # 応答候補に追加
    cdd = ResponseCandidate(output, 0.7 + random.random())
    candidateList.append(cdd)  
       
def Calc(text):
    fixs = {"を計算して":"","たす":"+","足す":"+","＋":"+","ひく":"-","引く":"-","ー":"-","かける":"*","かける":"*","掛ける":"*",
            "×":"*","x":"*","X":"*","＊":"*","わる":"/","割る":"/","÷":"/","（":"(","）":")",
            "１":"1","２":"2","３":"3","４":"4","５":"5","６":"6","７":"7","８":"8","９":"9","０":"0","　":" ",
            "一":"1","二":"2","三":"3","四":"4","五":"5","六":"6","七":"7","八":"8","九":"9","零":"0"}
    for i,j in fixs.items():
        text= text.replace(i,j)
    print(text)
    try:
        ans = eval(text)
        print(type(ans))
        if type(ans)==float:
            return "その答えは"+str(eval(text))+"...です。"
        else:
            return "その答えは"+str(eval(text))+"です。"
    except:
        return "無効な入力です。"

import time
isRun = False
def Ramen():
    global isRun
    if isRun:
        return "しばしお待ちください…"
    isRun = True
    start = time.time()
    while time.time()-start < 180:
        pass
    isRun = False
    return "3分経ちました"

def brainfxxk(text):
    code = list()
    text1,text2 = text.split("入力:")
    text1 = text1.replace('&gt','>')
    text1 = text1.replace('&lt','<')
    print(text2)
    for c in text1:
        if c=='+' or  c=='-' or  c=='>' or  c=='<' or  c==',' or  c=='.' or  c=='[' or c==']':
            code.append(c)
    codesize = len(code)
    memory = [0]
    pointer = 0
    cpointer = 0
    index = 0
    ans = ""
    while cpointer<codesize:
        if cpointer<0:
            return "エラーが発生しました。"
        if code[cpointer]=='+':
            memory[pointer] += 1
            if memory[pointer]==256:
                memory[pointer]= 0
        if code[cpointer]=='-':
            memory[pointer] -= 1
            if memory[pointer]==-1:
                memory[pointer]= 255
        if code[cpointer]=='<':
            pointer -= 1
            if pointer<0:
                return "エラーが発生しました。"
        if code[cpointer]=='>':
            pointer += 1
            if pointer==len(memory):
                memory.append(0)
        if code[cpointer]==',':
            if index>=len(text2):
                memory[pointer] = 0
            else:
                memory[pointer] = ord(text2[index])
                index += 1
        if code[cpointer]=='.':
            ans += chr(memory[pointer])
            print(chr(memory[pointer]))
        if code[cpointer]=='[':
            if memory[pointer]==0:
                loopCounter = 1
                while loopCounter > 0:
                    cpointer += 1
                    if code[cpointer]=='[':
                        loopCounter += 1
                    if code[cpointer]==']':
                        loopCounter -= 1
        if code[cpointer]==']':
            if memory[pointer]!=0:
                loopCounter = 1
                while loopCounter > 0:
                    cpointer -= 1
                    if code[cpointer]=='[':
                        loopCounter -= 1
                    if code[cpointer]==']':
                        loopCounter += 1
        cpointer += 1
    return ans

# 応答文を生成する関数
def generateResponse(inputText):
    
    # 応答文候補を空にしておく
    candidateList.clear()
    
    # 形態素解析した後，3つの戦略を順番に実行
    wordlist = janomeAnalyzer(inputText)
    generateResponseByRule(inputText)
    generateResponseByInputTopic(wordlist)
    generateOtherResponse()
    
    # ネガポジ判定の結果を応答候補に追加
    generateNegaposiResponse(inputText)


    
    ret="デフォルト"
    maxScore=-1.0

    # scoreが最も高い応答文候補を戻す
    for cdd in candidateList:
        cdd.print()
        if cdd.score > maxScore:
            ret=cdd.response
            maxScore = cdd.score
    return ret
###################################################

# キーワード照合ルールを読み込む
setupKeywordMatchingRule()

# 特定の文字列に対して返答
@respond_to('こんにちは')
def respond(message):
    message.reply('こんにちは！')

# デフォルトの返答
@default_reply()
def default(message):
    # Slackの入力を取得
    text = message.body['text']

    # システムの出力を生成
    output = generateResponse(text)

    # Slackで返答
    message.reply(output)
    
# スタンプの追加
@respond_to('かっこいい')
def react(message):
    message.reply('ありがとう！')
    message.react('hearts')
    message.react('+1')

@respond_to("を計算して")
def respond(message):
    text = Calc(message.body['text'])
    message.reply(text)

@respond_to("3分測って")
def respond(message):
    if not isRun:
        message.reply("3分間待ってやる")
        message.react('sunglasses')
    text = Ramen()
    message.reply(text)

@respond_to("コマンド:")
def respond(message):
    output = brainfxxk(message.body['text'])
    message.reply("実行結果: "+output)