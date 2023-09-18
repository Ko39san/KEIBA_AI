#基本ライブラリ
import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import requests
from bs4 import BeautifulSoup
import time
import re
from urllib.request import urlopen
import optuna.integration.lightgbm as lgb_o
from itertools import combinations, permutations
import matplotlib.pyplot as plt
from io import StringIO
import datetime



class DataProcessor:
    """
    Attributes:
    ----------
    data : pd.DataFrame
        rawデータ
    data_p : pd.DataFrame
        preprocessing後のデータ
    data_h : pd.DataFrame
        merge_horse_results後のデータ
    data_pe : pd.DataFrame
        merge_peds後のデータ
    data_c : pd.DataFrame
        process_categorical後のデータ
    no_peds: Numpy.array
        merge_pedsを実行した時に、血統データが存在しなかった馬のhorse_id一覧
    """

    def __init__(self):
        self.data = pd.DataFrame()
        self.data_p = pd.DataFrame()
        self.data_h = pd.DataFrame()
        self.data_pe = pd.DataFrame()
        self.data_c = pd.DataFrame()

    def merge_horse_results(self, hr, n_samples_list=[5, 9, 'all']):
        """
        馬の過去成績データから、
        n_samples_listで指定されたレース分の着順と賞金の平均を追加してdata_hに返す

        Parameters:
        ----------
        hr : HorseResults
            馬の過去成績データ
        n_samples_list : list, default [5, 9, 'all']
            過去何レース分追加するか
        """

        self.data_h = self.data_p.copy()
        for n_samples in n_samples_list:
            self.data_h = hr.merge_all(self.data_h, n_samples=n_samples)

        #6/6追加： 馬の出走間隔追加
        self.data_h['interval'] = (self.data_h['date'] - self.data_h['latest']).dt.days
        self.data_h.drop(['開催', 'latest'], axis=1, inplace=True)

    def merge_peds(self, peds):
        """
        5世代分血統データを追加してdata_peに返す

        Parameters:
        ----------
        peds : Peds.peds_e
            Pedsクラスで加工された血統データ。
        """

        self.data_pe = \
            self.data_h.merge(peds, left_on='horse_id', right_index=True,
                                                             how='left')
        self.no_peds = self.data_pe[self.data_pe['peds_0'].isnull()]\
            ['horse_id'].unique()
        if len(self.no_peds) > 0:
            print('scrape peds at horse_id_list "no_peds"')

    def process_categorical(self, le_horse, le_jockey, results_m):
        """
        カテゴリ変数を処理してdata_cに返す

        Parameters:
        ----------
        le_horse : sklearn.preprocessing.LabelEncoder
            horse_idを0始まりの整数に変換するLabelEncoderオブジェクト。
        le_jockey : sklearn.preprocessing.LabelEncoder
            jockey_idを0始まりの整数に変換するLabelEncoderオブジェクト。
        results_m : Results.data_pe
            ダミー変数化のとき、ResultsクラスとShutubaTableクラスで列を合わせるためのもの
        """

        df = self.data_pe.copy()

        #ラベルエンコーディング。horse_id, jockey_idを0始まりの整数に変換
        mask_horse = df['horse_id'].isin(le_horse.classes_)
        new_horse_id = df['horse_id'].mask(mask_horse).dropna().unique()
        le_horse.classes_ = np.concatenate([le_horse.classes_, new_horse_id])
        df['horse_id'] = le_horse.transform(df['horse_id'])
        mask_jockey = df['jockey_id'].isin(le_jockey.classes_)
        new_jockey_id = df['jockey_id'].mask(mask_jockey).dropna().unique()
        le_jockey.classes_ = np.concatenate([le_jockey.classes_, new_jockey_id])
        df['jockey_id'] = le_jockey.transform(df['jockey_id'])

        #horse_id, jockey_idをpandasのcategory型に変換
        df['horse_id'] = df['horse_id'].astype('category')
        df['jockey_id'] = df['jockey_id'].astype('category')

        #そのほかのカテゴリ変数をpandasのcategory型に変換してからダミー変数化
        #列を一定にするため
        weathers = results_m['weather'].unique()
        race_types = results_m['race_type'].unique()
        ground_states = results_m['ground_state'].unique()
        sexes = results_m['性'].unique()
        df['weather'] = pd.Categorical(df['weather'], weathers)
        df['race_type'] = pd.Categorical(df['race_type'], race_types)
        df['ground_state'] = pd.Categorical(df['ground_state'], ground_states)
        df['性'] = pd.Categorical(df['性'], sexes)
        df = pd.get_dummies(df, columns=['weather', 'race_type', 'ground_state', '性'])

        self.data_c = df

class Results(DataProcessor):
    def __init__(self, results):
        super(Results, self).__init__()
        self.data = results

    @classmethod
    def read_pickle(cls, path_list):
        df = pd.read_pickle(path_list[0])
        for path in path_list[1:]:
            df = update_data(df, pd.read_pickle(path))
        return cls(df)

    @staticmethod
    def scrape(race_id_list):
        """
        レース結果データをスクレイピングする関数

        Parameters:
        ----------
        race_id_list : list
            レースIDのリスト

        Returns:
        ----------
        race_results_df : pandas.DataFrame
            全レース結果データをまとめてDataFrame型にしたもの
        """

        #race_idをkeyにしてDataFrame型を格納
        race_results = {}
        for race_id in (race_id_list):
            time.sleep(1)
            try:
                url = "https://db.netkeiba.com/race/" + race_id

                html = requests.get(url)
                html.encoding = "EUC-JP"

                #メインとなるテーブルデータを取得
                buffer = StringIO(html.text)
                df = pd.read_html(buffer)[0]
                # 列名に半角スペースがあれば除去する
                df = df.rename(columns=lambda x: x.replace(' ', ''))

                # 天候、レースの種類、コースの長さ、馬場の状態、日付をスクレイピング
                soup = BeautifulSoup(html.text, "html.parser")
                #天候、レースの種類、コースの長さ、馬場の状態、日付をスクレイピング
                texts = (
                    soup.find("div", attrs={"class": "data_intro"}).find_all("p")[0].text
                    + soup.find("div", attrs={"class": "data_intro"}).find_all("p")[1].text
                )
                info = re.findall(r'\w+', texts)
                for text in info:
                    if text in ["芝", "ダート"]:
                        df["race_type"] = [text] * len(df)
                    if "障" in text:
                        df["race_type"] = ["障害"] * len(df)
                    if "m" in text:
                        df["course_len"] = [int(re.findall(r"\d+", text)[-1])] * len(df) #20211212：[0]→[-1]に修正
                    if text in ["良", "稍重", "重", "不良"]:
                        df["ground_state"] = [text] * len(df)
                    if text in ["曇", "晴", "雨", "小雨", "小雪", "雪"]:
                        df["weather"] = [text] * len(df)
                    if "年" in text:
                        df["date"] = [text] * len(df)

                #馬ID、騎手IDをスクレイピング
                horse_id_list = []
                horse_a_list = soup.find("table", attrs={"summary": "レース結果"}).find_all(
                    "a", attrs={"href": re.compile("^/horse")}
                )
                for a in horse_a_list:
                    horse_id = re.findall(r"\d+", a["href"])
                    horse_id_list.append(horse_id[0])
                jockey_id_list = []
                jockey_a_list = soup.find("table", attrs={"summary": "レース結果"}).find_all(
                    "a", attrs={"href": re.compile("^/jockey")}
                )
                for a in jockey_a_list:
                    jockey_id = re.findall(r"\d+", a["href"])
                    jockey_id_list.append(jockey_id[0])
                df["horse_id"] = horse_id_list
                df["jockey_id"] = jockey_id_list

                #インデックスをrace_idにする
                df.index = [race_id] * len(df)

                race_results[race_id] = df
            #存在しないrace_idを飛ばす
            except IndexError:
                continue
            except AttributeError: #存在しないrace_idでAttributeErrorになるページもあるので追加
                continue
            #wifiの接続が切れた時などでも途中までのデータを返せるようにする
            except Exception as e:
                print(e)
                break
            #Jupyterで停止ボタンを押した時の対処
            except:
                break

        #pd.DataFrame型にして一つのデータにまとめる
        race_results_df = pd.concat([race_results[key] for key in race_results])
        race_results_df.columns = race_results_df.columns.str.replace(' ', '')
        # 列名に半角スペースがあれば除去する（全体データに対して）
        race_results_df = race_results_df.rename(columns=lambda x: x.replace(' ', ''))
        return race_results_df

    #前処理
    def preprocessing(self):
        df = self.data.copy()

        # 着順に数字以外の文字列が含まれているものを取り除く
        # 全角スペースを削除
        #df['着順'] = pd.to_numeric(df['着順'], errors='coerce')
        #df.dropna(subset=['着順'], inplace=True)
        #df['着順'] = df['着順'].astype(int)
        #df['rank'] = df['着順'].map(lambda x:1 if x<4 else 0)

        # 性齢を性と年齢に分ける
        df["性"] = df["性齢"].map(lambda x: str(x)[0])
        df["年齢"] = df["性齢"].map(lambda x: str(x)[1:]).astype(int)

        # 情報開示前の場合
        df["馬体重(増減)"].fillna('0(0)', inplace=True)

        # 馬体重を体重と体重変化に分ける
        df = df[df["馬体重(増減)"] != '--']
        df["体重"] = df["馬体重(増減)"].str.split("(", expand=True)[0].astype(int)
        df["体重変化"] = df["馬体重(増減)"].str.split("(", expand=True)[1].str[:-1]
        # 2020/12/13追加：増減が「前計不」などのとき欠損値にする
        df['体重変化'] = pd.to_numeric(df['体重変化'], errors='coerce')

        df["date"] = pd.to_datetime(df["date"])


        # 単勝をfloatに変換
        #df["単勝"] = df["単勝"].astype(float)
        # 距離は10の位を切り捨てる
        df["course_len"] = df["course_len"].astype(float) // 100

        # 不要な列を削除
        df.drop(["性齢", '馬名', '騎手', '人気'],
                axis=1, inplace=True)

        df["date"] = pd.to_datetime(df["date"], format="%Y年%m月%d日")

        #開催場所
        df['開催'] = df.index.map(lambda x:str(x)[4:6])

        #6/6出走数追加
        df['n_horses'] = df.index.map(df.index.value_counts())

        self.data_p = df

    #カテゴリ変数の処理
    def process_categorical(self):
        self.le_horse = LabelEncoder().fit(self.data_pe['horse_id'])
        self.le_jockey = LabelEncoder().fit(self.data_pe['jockey_id'])
        super().process_categorical(self.le_horse, self.le_jockey, self.data_pe)

class ShutubaTable(DataProcessor):
    def __init__(self, shutuba_tables):
        super(ShutubaTable, self).__init__()
        self.data = shutuba_tables

    @classmethod
    def scrape(cls, race_id_list, date):
        data = pd.DataFrame()
        for race_id in (race_id_list):
            time.sleep(1)
            url = 'https://race.netkeiba.com/race/shutuba.html?race_id=' + race_id

            html = requests.get(url)
            html.encoding = "EUC-JP"

            buffer = StringIO(html.text)
            df = pd.read_html(buffer)[0]

            # 列名に半角スペースがあれば除去する
            df = df.rename(columns=lambda x: x.replace(' ', ''))
            df = df.T.reset_index(level=0, drop=True).T

            soup = BeautifulSoup(html.text, "html.parser")

            texts = soup.find('div', attrs={'class': 'RaceData01'}).text
            texts = re.findall(r'\w+', texts)
            for text in texts:
                if 'm' in text:
                    df['course_len'] = [int(re.findall(r'\d+', text)[-1])] * len(df) #20211212：[0]→[-1]に修正
                if text in ["曇", "晴", "雨", "小雨", "小雪", "雪"]:
                    df["weather"] = [text] * len(df)
                if text in ["良", "稍重", "重"]:
                    df["ground_state"] = [text] * len(df)
                if '不' in text:
                    df["ground_state"] = ['不良'] * len(df)
                # 2020/12/13追加
                if '稍' in text:
                    df["ground_state"] = ['稍重'] * len(df)
                if '芝' in text:
                    df['race_type'] = ['芝'] * len(df)
                if '障' in text:
                    df['race_type'] = ['障害'] * len(df)
                if 'ダ' in text:
                    df['race_type'] = ['ダート'] * len(df)
            df['date'] = [date] * len(df)

            # horse_id
            horse_id_list = []
            horse_td_list = soup.find_all("td", attrs={'class': 'HorseInfo'})
            for td in horse_td_list:
                horse_id = re.findall(r'\d+', td.find('a')['href'])[0]
                horse_id_list.append(horse_id)
            # jockey_id
            jockey_id_list = []
            jockey_td_list = soup.find_all("td", attrs={'class': 'Jockey'})
            for td in jockey_td_list:
                jockey_id = re.findall(r'\d+', td.find('a')['href'])[0]
                jockey_id_list.append(jockey_id)
            df['horse_id'] = horse_id_list
            df['jockey_id'] = jockey_id_list

            df.index = [race_id] * len(df)
            data = pd.concat([data, df])
        return cls(data)

    #前処理
    def preprocessing(self):
        df = self.data.copy()

        df["性"] = df["性齢"].map(lambda x: str(x)[0])
        df["年齢"] = df["性齢"].map(lambda x: str(x)[1:]).astype(int)

        # 情報開示前の場合
        df["馬体重(増減)"].fillna('0(0)', inplace=True)

        # 馬体重を体重と体重変化に分ける
        df = df[df["馬体重(増減)"] != '--']
        df["体重"] = df["馬体重(増減)"].str.split("(", expand=True)[0].astype(int)
        df["体重変化"] = df["馬体重(増減)"].str.split("(", expand=True)[1].str[:-1]
        # 2020/12/13追加：増減が「前計不」などのとき欠損値にする
        df['体重変化'] = pd.to_numeric(df['体重変化'], errors='coerce')

        df["date"] = pd.to_datetime(df["date"])

        df['枠'] = df['枠'].astype(int)
        df['馬番'] = df['馬番'].astype(int)
        df['斤量'] = df['斤量'].astype(int)
        df['開催'] = df.index.map(lambda x:str(x)[4:6])

        #6/6出走数追加
        df['n_horses'] = df.index.map(df.index.value_counts())

        # 距離は10の位を切り捨てる
        df["course_len"] = df["course_len"].astype(float) // 100

        # 使用する列を選択
        df = df[['枠', '馬番', '斤量', 'course_len', 'weather','race_type',
        'ground_state', 'date', 'horse_id', 'jockey_id', '性', '年齢',
       '体重', '体重変化', '開催', 'n_horses']]

        self.data_p = df.rename(columns={'枠': '枠番'})

class HorseResults:
    def __init__(self, horse_results):
        self.horse_results = horse_results[['日付', '着順', '賞金', '着差', '通過', '開催', '距離']]
        self.preprocessing()

    @classmethod
    def read_pickle(cls, path_list):
        df = pd.read_pickle(path_list[0])
        for path in path_list[1:]:
            df = update_data(df, pd.read_pickle(path))
        return cls(df)




    @staticmethod
    def scrape(horse_id_list):
        horse_results = {}
        for horse_id in (horse_id_list):
            time.sleep(1)
            try:
                url = 'https://db.netkeiba.com/horse/' + horse_id
                res = requests.get(url)
                res.encoding = 'EUC-JP'  # またはサイトに合わせたエンコーディング
                df = pd.read_html(res.text)[3]
            
                if df.columns[0]=='受賞歴':
                    df = pd.read_html(url)[4]
                df.index = [horse_id] * len(df)
                horse_results[horse_id] = df
            except IndexError:
                st.error(f"IndexError occurred for horse_id: {horse_id}")
                continue
            except Exception as e:
                st.error(f"An exception occurred: {e}")
                break

        if not horse_results:
            st.error("horse_results is empty")
            return

        #pd.DataFrame型にして一つのデータにまとめる
        horse_results_df = pd.concat([horse_results[key] for key in horse_results])

        return horse_results_df

    def preprocessing(self):
        df = self.horse_results.copy()

        # 着順に数字以外の文字列が含まれているものを取り除く
        df['着順'] = pd.to_numeric(df['着順'], errors='coerce')
        df.dropna(subset=['着順'], inplace=True)
        df['着順'] = df['着順'].astype(int)

        df["date"] = pd.to_datetime(df["日付"])
        df.drop(['日付'], axis=1, inplace=True)

        #賞金のNaNを0で埋める
        df['賞金'].fillna(0, inplace=True)

        #1着の着差を0にする
        df['着差'] = df['着差'].map(lambda x: 0 if x<0 else x)

        #レース展開データ
        #n=1: 最初のコーナー位置, n=4: 最終コーナー位置
        def corner(x, n):
            if type(x) != str:
                return x
            elif n==4:
                return int(re.findall(r'\d+', x)[-1])
            elif n==1:
                return int(re.findall(r'\d+', x)[0])
        df['first_corner'] = df['通過'].map(lambda x: corner(x, 1))
        df['final_corner'] = df['通過'].map(lambda x: corner(x, 4))

        df['final_to_rank'] = df['final_corner'] - df['着順']
        df['first_to_rank'] = df['first_corner'] - df['着順']
        df['first_to_final'] = df['first_corner'] - df['final_corner']

        #開催場所
        df['開催'] = df['開催'].str.extract(r'(\D+)')[0].map(place_dict).fillna('11')
        #race_type
        df['race_type'] = df['距離'].str.extract(r'(\D+)')[0].map(race_type_dict)
        #距離は10の位を切り捨てる
        #一部の馬で欠損値があり、intに変換できないためfloatに変換する
        df['course_len'] = df['距離'].str.extract(r'(\d+)').astype(float) // 100
        df.drop(['距離'], axis=1, inplace=True)
        #インデックス名を与える
        df.index.name = 'horse_id'

        self.horse_results = df
        self.target_list = ['着順', '賞金', '着差', 'first_corner', 'final_corner',
                            'first_to_rank', 'first_to_final','final_to_rank']

    #n_samplesレース分馬ごとに平均する
    def average(self, horse_id_list, date, n_samples='all'):
        target_df = self.horse_results.query('index in @horse_id_list')

        #過去何走分取り出すか指定
        if n_samples == 'all':
            filtered_df = target_df[target_df['date'] < date]
        elif n_samples > 0:
            filtered_df = target_df[target_df['date'] < date].\
                sort_values('date', ascending=False).groupby(level=0).head(n_samples)
        else:
            raise Exception('n_samples must be >0')

        #集計して辞書型に入れる
        self.average_dict = {}
        self.average_dict['non_category'] = filtered_df.groupby(level=0)[self.target_list].mean()\
            .add_suffix('_{}R'.format(n_samples))
        for column in ['course_len', 'race_type', '開催']:
            self.average_dict[column] = filtered_df.groupby(['horse_id', column])\
                [self.target_list].mean().add_suffix('_{}_{}R'.format(column, n_samples))

        #6/6追加: 馬の出走間隔追加のために、全レースの日付を変数latestに格納
        if n_samples == 5:
            self.latest = filtered_df.groupby('horse_id')['date'].max().rename('latest')

    def merge(self, results, date, n_samples='all'):
        df = results[results['date']==date]
        horse_id_list = df['horse_id']
        self.average(horse_id_list, date, n_samples)
        merged_df = df.merge(self.average_dict['non_category'], left_on='horse_id',
                             right_index=True, how='left')
        for column in ['course_len','race_type', '開催']:
            merged_df = merged_df.merge(self.average_dict[column],
                                        left_on=['horse_id', column],
                                        right_index=True, how='left')

        #6/6追加：馬の出走間隔追加のために、全レースの日付を変数latestに格納
        if n_samples == 5:
            merged_df = merged_df.merge(self.latest, left_on='horse_id',
                             right_index=True, how='left')
        return merged_df

    def merge_all(self, results, n_samples='all'):
        date_list = results['date'].unique()
        merged_df = pd.concat([self.merge(results, date, n_samples) for date in (date_list)])
        return merged_df

class Peds:
    def __init__(self, peds):
        self.peds = peds
        self.peds_e = pd.DataFrame() #after label encoding and transforming into category

    @classmethod
    def read_pickle(cls, path_list):
        df = pd.read_pickle(path_list[0])
        for path in path_list[1:]:
            df = update_data(df, pd.read_pickle(path))
        return cls(df)


    @staticmethod
    def scrape(horse_id_list):
        peds_dict = {}

        # horse_id_listが空でないか確認
        if len(horse_id_list) == 0:
            print("No horse IDs to scrape.")
            return pd.DataFrame()

        for horse_id in (horse_id_list):
            time.sleep(1)
            try:
                url = "https://db.netkeiba.com/horse/ped/" + horse_id
                html = requests.get(url)
                html.encoding = "EUC-JP"
                df = pd.read_html(html.text)[0]

                generations = {}
                for i in reversed(range(5)):
                    generations[i] = df[i]
                    df.drop([i], axis=1, inplace=True)
                    df = df.drop_duplicates()
                ped = pd.concat([generations[i] for i in range(5)]).rename(horse_id)
                peds_dict[horse_id] = ped.reset_index(drop=True)

            except IndexError:
                continue
            except Exception as e:
                print(e)
                break
            except:
                break

        if len(peds_dict) == 0:
            print("No data to concatenate. Skipping...")
            return pd.DataFrame()

        peds_df = pd.concat([peds_dict[key] for key in peds_dict], axis=1).T.add_prefix('peds_')

        return peds_df



    def encode(self):
        df = self.peds.copy()
        for column in df.columns:
            df[column] = LabelEncoder().fit_transform(df[column].fillna('Na'))
        self.peds_e = df.astype('category')

        
#開催場所をidに変換するための辞書型
place_dict = {
    '札幌':'01',  '函館':'02',  '福島':'03',  '新潟':'04',  '東京':'05',
    '中山':'06',  '中京':'07',  '京都':'08',  '阪神':'09',  '小倉':'10'
}

#レースタイプをレース結果データと整合させるための辞書型
race_type_dict = {
    '芝': '芝', 'ダ': 'ダート', '障': '障害'
}



# データをロードする関数
def update_data(old, new):
    """
    Parameters:
    ----------
    old : pandas.DataFrame
        古いデータ
    new : pandas.DataFrame
        新しいデータ
    """

    filtered_old = old[~old.index.isin(new.index)]
    return pd.concat([filtered_old, new])

def load_data(base_race_id):
    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={base_race_id}"
    html = requests.get(url)
    html.encoding = "EUC-JP"

    try:
        html_text = html.text
        buffer = StringIO(html_text)
        df = pd.read_html(buffer)[0]
        df.columns = df.iloc[0]
        column_names = ['枠', '馬 番', '印', '馬名', '性齢', '斤量', '騎手', '厩舎', '馬体重 (増減)', '予想オッズ', '人気', '登録', 'メモ']
        df.columns = column_names
        df.drop(['印', '登録', 'メモ', '予想オッズ', '人気'], axis=1, inplace=True)
        return df
    except Exception as e:
        st.write(f"エラーが発生しました: {e}")
        return None

def load_additional_data(base_race_id):
    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={base_race_id}"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    try:
        race_name_div = soup.find('div', class_='RaceName')
        race_name = race_name_div.text.strip()

        # Icon_GradeType13 クラスを持つ span タグが存在するか確認
        if race_name_div.find('span', class_='Icon_GradeType13'):
            race_name += '_win5'

        race_data01 = soup.find('div', class_='RaceData01').text.strip()
        race_data02 = soup.find('div', class_='RaceData02').text.strip().replace('\xa0', ' ')

        return {
            'race_name': race_name,
            'race_data01': race_data01,
            'race_data02': race_data02
        }
    except Exception as e:
        st.write(f"追加データの取得中にエラーが発生しました: {e}")
        return None

def generate_column_names():
    column_names = ['枠番',
 '馬番',
 '斤量',
 'course_len',
 'horse_id',
 'jockey_id',
 '年齢',
 '体重',
 '体重変化',
 'n_horses',
 '着順_5R',
 '賞金_5R',
 '着差_5R',
 'first_corner_5R',
 'final_corner_5R',
 'first_to_rank_5R',
 'first_to_final_5R',
 'final_to_rank_5R',
 '着順_course_len_5R',
 '賞金_course_len_5R',
 '着差_course_len_5R',
 'first_corner_course_len_5R',
 'final_corner_course_len_5R',
 'first_to_rank_course_len_5R',
 'first_to_final_course_len_5R',
 'final_to_rank_course_len_5R',
 '着順_race_type_5R',
 '賞金_race_type_5R',
 '着差_race_type_5R',
 'first_corner_race_type_5R',
 'final_corner_race_type_5R',
 'first_to_rank_race_type_5R',
 'first_to_final_race_type_5R',
 'final_to_rank_race_type_5R',
 '着順_開催_5R',
 '賞金_開催_5R',
 '着差_開催_5R',
 'first_corner_開催_5R',
 'final_corner_開催_5R',
 'first_to_rank_開催_5R',
 'first_to_final_開催_5R',
 'final_to_rank_開催_5R',
 '着順_9R',
 '賞金_9R',
 '着差_9R',
 'first_corner_9R',
 'final_corner_9R',
 'first_to_rank_9R',
 'first_to_final_9R',
 'final_to_rank_9R',
 '着順_course_len_9R',
 '賞金_course_len_9R',
 '着差_course_len_9R',
 'first_corner_course_len_9R',
 'final_corner_course_len_9R',
 'first_to_rank_course_len_9R',
 'first_to_final_course_len_9R',
 'final_to_rank_course_len_9R',
 '着順_race_type_9R',
 '賞金_race_type_9R',
 '着差_race_type_9R',
 'first_corner_race_type_9R',
 'final_corner_race_type_9R',
 'first_to_rank_race_type_9R',
 'first_to_final_race_type_9R',
 'final_to_rank_race_type_9R',
 '着順_開催_9R',
 '賞金_開催_9R',
 '着差_開催_9R',
 'first_corner_開催_9R',
 'final_corner_開催_9R',
 'first_to_rank_開催_9R',
 'first_to_final_開催_9R',
 'final_to_rank_開催_9R',
 '着順_allR',
 '賞金_allR',
 '着差_allR',
 'first_corner_allR',
 'final_corner_allR',
 'first_to_rank_allR',
 'first_to_final_allR',
 'final_to_rank_allR',
 '着順_course_len_allR',
 '賞金_course_len_allR',
 '着差_course_len_allR',
 'first_corner_course_len_allR',
 'final_corner_course_len_allR',
 'first_to_rank_course_len_allR',
 'first_to_final_course_len_allR',
 'final_to_rank_course_len_allR',
 '着順_race_type_allR',
 '賞金_race_type_allR',
 '着差_race_type_allR',
 'first_corner_race_type_allR',
 'final_corner_race_type_allR',
 'first_to_rank_race_type_allR',
 'first_to_final_race_type_allR',
 'final_to_rank_race_type_allR',
 '着順_開催_allR',
 '賞金_開催_allR',
 '着差_開催_allR',
 'first_corner_開催_allR',
 'final_corner_開催_allR',
 'first_to_rank_開催_allR',
 'first_to_final_開催_allR',
 'final_to_rank_開催_allR',
 'interval',
 'peds_0',
 'peds_1',
 'peds_2',
 'peds_3',
 'peds_4',
 'peds_5',
 'peds_6',
 'peds_7',
 'peds_8',
 'peds_9',
 'peds_10',
 'peds_11',
 'peds_12',
 'peds_13',
 'peds_14',
 'peds_15',
 'peds_16',
 'peds_17',
 'peds_18',
 'peds_19',
 'peds_20',
 'peds_21',
 'peds_22',
 'peds_23',
 'peds_24',
 'peds_25',
 'peds_26',
 'peds_27',
 'peds_28',
 'peds_29',
 'peds_30',
 'peds_31',
 'peds_32',
 'peds_33',
 'peds_34',
 'peds_35',
 'peds_36',
 'peds_37',
 'peds_38',
 'peds_39',
 'peds_40',
 'peds_41',
 'peds_42',
 'peds_43',
 'peds_44',
 'peds_45',
 'peds_46',
 'peds_47',
 'peds_48',
 'peds_49',
 'peds_50',
 'peds_51',
 'peds_52',
 'peds_53',
 'peds_54',
 'peds_55',
 'peds_56',
 'peds_57',
 'peds_58',
 'peds_59',
 'peds_60',
 'peds_61',
 'weather_晴',
 'weather_曇',
 'weather_雨',
 'weather_小雨',
 'weather_小雪',
 'weather_雪',
 'race_type_芝',
 'race_type_ダート',
 'race_type_障害',
 'ground_state_良',
 'ground_state_稍重',
 'ground_state_重',
 'ground_state_不良',
 '性_牝',
 '性_牡',
 '性_セ']
    return column_names


# Streamlit UI
st.title("競馬AI予想🐎")

# 現在の日付をデフォルトとして設定
today = datetime.date.today()
# date_input ウィジェットで日付を選択
selected_date = st.date_input("開催日を選択してください", today)
# 選択された日付を YYYY/MM/DD 形式で表示
formatted_date = selected_date.strftime("%Y/%m/%d")

racecourse_map = {
    "札幌_01": "01",
    "函館_02": "02",
    "福島_03": "03",
    "新潟_04": "04",
    "東京_05": "05",
    "中山_06": "06",
    "中京_07": "07",
    "京都_08": "08",
    "阪神_09": "09",
    "小倉_10": "10"
}

racecourse = st.selectbox("競馬場を選択してください", list(racecourse_map.keys()))
holding_number = st.selectbox("開催回数を選択してください", list(range(1, 12)))

day_options = ["1日目", "2日目", "3日目", "4日目", "5日目", "6日目", "7日目", "8日目", "9日目", "10日目"]
selected_day = st.selectbox("何日目か選択してください", day_options)
day_number = int(selected_day[0])

race_number = st.selectbox("何レースかを選択してください", list(range(1, 12)))



base_race_id = f"2023{racecourse_map[racecourse]}{holding_number:02d}{day_number:02d}{race_number:02d}"


st.write(f"選択された日付は {formatted_date} です")
st.write(f"RACE_IDは {base_race_id} です。")


# データをロード
df = load_data(base_race_id)
additional_data = load_additional_data(base_race_id)

if additional_data:
    st.write(f"レース名: {additional_data['race_name']}")


# DataFrameを表示
if df is not None:
    st.table(df)
else:
    st.write('データをロードできませんでした。')






if st.button('AI予想'):
    st.write('AI予想を開始致します。処理には15分〜20分かかります。')

    #race_id_list の生成
    #race_id_list = [f"{2023010101}{str(i).zfill(2)}" for i in range(1, 13)]
    race_id_list = [base_race_id]
    sta = ShutubaTable.scrape(race_id_list, formatted_date)
    sta.data = sta.data.rename(columns=lambda x: x.replace(' ', ''))
    horse_id_list = sta.data['horse_id'].unique()
    #前処理
    sta.preprocessing()
    st.write("出馬表: ", sta.data)
    
    horse_results = HorseResults.scrape(horse_id_list)
    

    horse_results = horse_results.rename(columns=lambda x: x.replace(' ', ''))
    st.write("出走馬の過去成績情報: ", horse_results)
    #馬の過去成績データ追加
    hr = HorseResults(horse_results)
    #馬の過去成績データの追加。新馬はNaNが追加される
    sta.merge_horse_results(hr)



    peds = Peds.scrape(horse_id_list)
    st.write("出走馬の血統情報: ", peds)

    p = Peds(peds)
    p.encode()

    sta.merge_horse_results(hr, n_samples_list=[5, 9, 'all'])

    #5世代分の血統データの追加
    sta.merge_peds(p.peds_e)

    data_pe = sta.data_pe

    # 1. LabelEncoderオブジェクトを初期化
    le_horse = LabelEncoder().fit(data_pe['horse_id'])
    le_jockey = LabelEncoder().fit(data_pe['jockey_id'])

    # 2. ラベルエンコーディング
    data_pe['horse_id'] = le_horse.transform(data_pe['horse_id'])
    data_pe['jockey_id'] = le_jockey.transform(data_pe['jockey_id'])

    # 3. pandasのcategory型に変換
    data_pe['horse_id'] = data_pe['horse_id'].astype('category')
    data_pe['jockey_id'] = data_pe['jockey_id'].astype('category')

    # 4. ダミー変数化
    weathers = data_pe['weather'].unique()
    race_types = data_pe['race_type'].unique()
    ground_states = data_pe['ground_state'].unique()
    sexes = data_pe['性'].unique()

    data_pe['weather'] = pd.Categorical(data_pe['weather'], weathers)
    data_pe['race_type'] = pd.Categorical(data_pe['race_type'], race_types)
    data_pe['ground_state'] = pd.Categorical(data_pe['ground_state'], ground_states)
    data_pe['性'] = pd.Categorical(data_pe['性'], sexes)

    data_pe = pd.get_dummies(data_pe, columns=['weather', 'race_type', 'ground_state', '性'])

    st.write("5世代分の血統データの追加: ", data_pe)


    # LightGBMモデルを読み込む
    lgb_clf = lgb.Booster(model_file="lgb_model.txt")

    data_c = data_pe.drop(['date'], axis=1)


    # 追加したい列名のリスト
    columns_to_add = ['ground_state_稍重', 'weather_雪', 'weather_小雨', 'ground_state_不良', 'weather_小雪', 'weather_雨', 'weather_曇', 'ground_state_重', 'race_type_障害', 'weather_晴', 'race_type_芝', 'race_type_ダート', 'ground_state_良', '性_牡', '性_牝', '性_セ']
    # データフレームに存在しない列名だけを追加
    for col in columns_to_add:
        if col not in data_c.columns:
            data_c[col] = 0  # 数値の0を入れる

    my_column_names = generate_column_names()

    # 訓練時に使用された特徴量の名前を取得
    train_features = my_column_names

    # 予測データの特徴量の名前を取得
    test_features = data_c.columns.tolist()

    # 訓練データにはあるが、予測データにはない特徴量を見つける
    missing_features = set(train_features) - set(test_features)

    # 不足している特徴量に0を割り当てる
    for feature in missing_features:
        data_c[feature] = 0


    # 予測を実施
    predictions = lgb_clf.predict(data_c)

    # 予測結果をdata_cに追加
    data_c['Predicted_Rank'] = predictions

    # 予測結果を降順にソート
    sorted_predictions = data_c.sort_values(by=['Predicted_Rank'], ascending=False)

    # TOP3の予測結果を抽出
    top_3_per_race = data_c.groupby(level=0).apply(lambda x: x.nlargest(3, 'Predicted_Rank'))

    # 各レースごとに出馬表とTOP3を表示
    for race_id, group_data in sorted_predictions.groupby(level=0):

        # df（出馬表）を対応するrace_idで更新する
        df = load_data(race_id)

        if df is not None:
            # 先頭に'AI予想'列を追加（すでに存在している場合はそのまま）
            if 'AI予想' not in df.columns:
                df.insert(0, 'AI予想', None)

            # race_idと馬番が一致する行の'AI予想'列にPredicted_Rankの値を挿入
            for _, row in group_data.iterrows():
                horse_number = row['馬番']
                predicted_rank = row['Predicted_Rank']
                df.loc[df['馬 番'] == horse_number, 'AI予想'] = predicted_rank

            # 追加の情報とともに出馬表を表示
            additional_data = load_additional_data(race_id)
            if additional_data:
                st.markdown(f"**レース名:** {additional_data['race_name']}")

            st.dataframe(df)

        # そのレースのTOP3予測を表示
        if race_id in top_3_per_race.index.levels[0]:
            top_3_data = top_3_per_race.loc[race_id]
            st.dataframe(top_3_data[['馬番', 'Predicted_Rank']])

