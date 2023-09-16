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

print("Current Directory:", os.getcwd())

class Results_1:
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
                # スクレイピング
                html = requests.get(url)
                html.encoding = "EUC-JP"
                # メインとなるテーブルデータを取得
                df = pd.read_html(html.text)[0]
                # 天候、レースの種類、コースの長さ、馬場の状態、日付をスクレイピング
                soup = BeautifulSoup(html.text, "html.parser")
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
                        df["course_len"] = [int(re.findall(r"\d+", text)[-1])] * len(df)
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
        return race_results_df

#馬の過去成績データを処理するクラス
class HorseResults_1:
    @staticmethod
    def scrape(horse_id_list):
        """
        馬の過去成績データをスクレイピングする関数

        Parameters:
        ----------
        horse_id_list : list
            馬IDのリスト

        Returns:
        ----------
        horse_results_df : pandas.DataFrame
            全馬の過去成績データをまとめてDataFrame型にしたもの
        """

        #horse_idをkeyにしてDataFrame型を格納
        horse_results = {}
        for horse_id in (horse_id_list):
            time.sleep(1)
            try:
                url = 'https://db.netkeiba.com/horse/' + horse_id
                df = pd.read_html(url)[3]
                #受賞歴がある馬の場合、3番目に受賞歴テーブルが来るため、4番目のデータを取得する
                if df.columns[0]=='受賞歴':
                    df = pd.read_html(url)[4]
                df.index = [horse_id] * len(df)
                horse_results[horse_id] = df
            except IndexError:
                continue
            except Exception as e:
                print(e)
                break
            except:
                break

        #pd.DataFrame型にして一つのデータにまとめる
        horse_results_df = pd.concat([horse_results[key] for key in horse_results])

        return horse_results_df

#払い戻し表データを処理するクラス
class Return:
    @staticmethod
    def scrape(race_id_list):
        """
        払い戻し表データをスクレイピングする関数

        Parameters:
        ----------
        race_id_list : list
            レースIDのリスト

        Returns:
        ----------
        return_tables_df : pandas.DataFrame
            全払い戻し表データをまとめてDataFrame型にしたもの
        """

        return_tables = {}
        for race_id in (race_id_list):
            time.sleep(1)
            try:
                url = "https://db.netkeiba.com/race/" + race_id

                #普通にスクレイピングすると複勝やワイドなどが区切られないで繋がってしまう。
                #そのため、改行コードを文字列brに変換して後でsplitする
                f = urlopen(url)
                html = f.read()
                html = html.replace(b'<br />', b'br')
                dfs = pd.read_html(html)

                #dfsの1番目に単勝〜馬連、2番目にワイド〜三連単がある
                df = pd.concat([dfs[1], dfs[2]])

                df.index = [race_id] * len(df)
                return_tables[race_id] = df
            except IndexError:
                continue
            except AttributeError: #存在しないrace_idでAttributeErrorになるページもあるので追加
                continue
            except Exception as e:
                print(e)
                break
            except:
                break

        #pd.DataFrame型にして一つのデータにまとめる
        return_tables_df = pd.concat([return_tables[key] for key in return_tables])
        return return_tables_df

    @staticmethod
    def scrape(race_id_list):
        """
        払い戻し表データをスクレイピングする関数

        Parameters:
        ----------
        race_id_list : list
            レースIDのリスト

        Returns:
        ----------
        return_tables_df : pandas.DataFrame
            全払い戻し表データをまとめてDataFrame型にしたもの
        """

        return_tables = {}
        for race_id in (race_id_list):
            time.sleep(1)
            try:
                url = "https://db.netkeiba.com/race/" + race_id

                #普通にスクレイピングすると複勝やワイドなどが区切られないで繋がってしまう。
                #そのため、改行コードを文字列brに変換して後でsplitする
                f = urlopen(url)
                html = f.read()
                html = html.replace(b'<br />', b'br')
                dfs = pd.read_html(html)

                #dfsの1番目に単勝〜馬連、2番目にワイド〜三連単がある
                df = pd.concat([dfs[1], dfs[2]])

                df.index = [race_id] * len(df)
                return_tables[race_id] = df
            except IndexError:
                continue
            except AttributeError: #存在しないrace_idでAttributeErrorになるページもあるので追加
                continue
            except Exception as e:
                print(e)
                break
            except:
                break

        #pd.DataFrame型にして一つのデータにまとめる
        return_tables_df = pd.concat([return_tables[key] for key in return_tables])
        return return_tables_df

#馬の過去成績データを処理するクラス

    @staticmethod
    def scrape(horse_id_list):
        """
        馬の過去成績データをスクレイピングする関数

        Parameters:
        ----------
        horse_id_list : list
            馬IDのリスト

        Returns:
        ----------
        horse_results_df : pandas.DataFrame
            全馬の過去成績データをまとめてDataFrame型にしたもの
        """

        #horse_idをkeyにしてDataFrame型を格納
        horse_results = {}
        for horse_id in (horse_id_list):
            time.sleep(1)
            try:
                url = 'https://db.netkeiba.com/horse/' + horse_id
                df = pd.read_html(url)[3]
                #受賞歴がある馬の場合、3番目に受賞歴テーブルが来るため、4番目のデータを取得する
                if df.columns[0]=='受賞歴':
                    df = pd.read_html(url)[4]
                df.index = [horse_id] * len(df)
                horse_results[horse_id] = df
            except IndexError:
                continue
            except Exception as e:
                print(e)
                break
            except:
                break

        #pd.DataFrame型にして一つのデータにまとめる
        horse_results_df = pd.concat([horse_results[key] for key in horse_results])

        return horse_results_df

#血統データを処理するクラス
class Peds_1:
    @staticmethod
    def scrape(horse_id_list):
        """
        血統データをスクレイピングする関数

        Parameters:
        ----------
        horse_id_list : list
            馬IDのリスト

        Returns:
        ----------
        peds_df : pandas.DataFrame
            全血統データをまとめてDataFrame型にしたもの
        """

        peds_dict = {}
        for horse_id in (horse_id_list):
            time.sleep(1)
            try:
                url = "https://db.netkeiba.com/horse/ped/" + horse_id
                df = pd.read_html(url)[0]

                #重複を削除して1列のSeries型データに直す
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

        #列名をpeds_0, ..., peds_61にする
        peds_df = pd.concat([peds_dict[key] for key in peds_dict], axis=1).T.add_prefix('peds_')

        return peds_df
    @staticmethod
    def scrape(horse_id_list):
        """
        血統データをスクレイピングする関数

        Parameters:
        ----------
        horse_id_list : list
            馬IDのリスト

        Returns:
        ----------
        peds_df : pandas.DataFrame
            全血統データをまとめてDataFrame型にしたもの
        """

        peds_dict = {}
        for horse_id in (horse_id_list):
            time.sleep(1)
            try:
                url = "https://db.netkeiba.com/horse/ped/" + horse_id
                df = pd.read_html(url)[0]

                #重複を削除して1列のSeries型データに直す
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

        #列名をpeds_0, ..., peds_61にする
        peds_df = pd.concat([peds_dict[key] for key in peds_dict], axis=1).T.add_prefix('peds_')

        return peds_df


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

class Results_2(DataProcessor):
    def __init__(self, results):
        super(Results_2, self).__init__()
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
                df = pd.read_html(html.text)[0]
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
        df['着順'] = pd.to_numeric(df['着順'], errors='coerce')
        df.dropna(subset=['着順'], inplace=True)
        df['着順'] = df['着順'].astype(int)
        df['rank'] = df['着順'].map(lambda x:1 if x<4 else 0)

        # 性齢を性と年齢に分ける
        df["性"] = df["性齢"].map(lambda x: str(x)[0])
        df["年齢"] = df["性齢"].map(lambda x: str(x)[1:]).astype(int)

        # 馬体重を体重と体重変化に分ける
        df["体重"] = df["馬体重"].str.split("(", expand=True)[0]
        df["体重変化"] = df["馬体重"].str.split("(", expand=True)[1].str[:-1]

        #errors='coerce'で、"計不"など変換できない時に欠損値にする
        df['体重'] = pd.to_numeric(df['体重'], errors='coerce')
        df['体重変化'] = pd.to_numeric(df['体重変化'], errors='coerce')

        # 単勝をfloatに変換
        df["単勝"] = df["単勝"].astype(float)
        # 距離は10の位を切り捨てる
        df["course_len"] = df["course_len"].astype(float) // 100

        # 不要な列を削除
        df.drop(["タイム", "着差", "調教師", "性齢", "馬体重", '馬名', '騎手', '人気', '着順'],
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

            df = pd.read_html(html.text)[0]
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
        try:
            df = self.data.copy()

            df["性"] = df["性齢"].map(lambda x: str(x)[0])
            df["年齢"] = df["性齢"].map(lambda x: str(x)[1:]).astype(int)

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
        except KeyError as e:
            st.write(f"エラーが発生しました: {e}")
            st.write("存在する列名:")
            st.write(self.data.columns)


class HorseResults_2:
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
        """
        馬の過去成績データをスクレイピングする関数

        Parameters:
        ----------
        horse_id_list : list
            馬IDのリスト

        Returns:
        ----------
        horse_results_df : pandas.DataFrame
            全馬の過去成績データをまとめてDataFrame型にしたもの
        """

        #horse_idをkeyにしてDataFrame型を格納
        horse_results = {}
        for horse_id in (horse_id_list):
            time.sleep(1)
            try:
                url = 'https://db.netkeiba.com/horse/' + horse_id
                df = pd.read_html(url)[3]
                #受賞歴がある馬の場合、3番目に受賞歴テーブルが来るため、4番目のデータを取得する
                if df.columns[0]=='受賞歴':
                    df = pd.read_html(url)[4]
                df.index = [horse_id] * len(df)
                horse_results[horse_id] = df
            except IndexError:
                continue
            except Exception as e:
                print(e)
                break
            except:
                break

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

#開催場所をidに変換するための辞書型
place_dict = {
    '札幌':'01',  '函館':'02',  '福島':'03',  '新潟':'04',  '東京':'05',
    '中山':'06',  '中京':'07',  '京都':'08',  '阪神':'09',  '小倉':'10'
}

#レースタイプをレース結果データと整合させるための辞書型
race_type_dict = {
    '芝': '芝', 'ダ': 'ダート', '障': '障害'
}

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
        """
        血統データをスクレイピングする関数

        Parameters:
        ----------
        horse_id_list : list
            馬IDのリスト

        Returns:
        ----------
        peds_df : pandas.DataFrame
            全血統データをまとめてDataFrame型にしたもの
        """

        peds_dict = {}
        for horse_id in (horse_id_list):
            time.sleep(1)
            try:
                url = "https://db.netkeiba.com/horse/ped/" + horse_id
                df = pd.read_html(url)[0]

                #重複を削除して1列のSeries型データに直す
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

        #列名をpeds_0, ..., peds_61にする
        peds_df = pd.concat([peds_dict[key] for key in peds_dict], axis=1).T.add_prefix('peds_')

        return peds_df

    def encode(self):
        df = self.peds.copy()
        for column in df.columns:
            df[column] = LabelEncoder().fit_transform(df[column].fillna('Na'))
        self.peds_e = df.astype('category')

class Return:
    def __init__(self, return_tables):
        self.return_tables = return_tables

    @classmethod
    def read_pickle(cls, path_list):
        df = pd.read_pickle(path_list[0])
        for path in path_list[1:]:
            df = update_data(df, pd.read_pickle(path))
        return cls(df)

    @staticmethod
    def scrape(race_id_list):
        return_tables = {}
        for race_id in (race_id_list):
            time.sleep(1)
            try:
                url = "https://db.netkeiba.com/race/" + race_id

                #普通にスクレイピングすると複勝やワイドなどが区切られないで繋がってしまう。
                #そのため、改行コードを文字列brに変換して後でsplitする
                f = urlopen(url)
                html = f.read()
                html = html.replace(b'<br />', b'br')
                dfs = pd.read_html(html)

                #dfsの1番目に単勝〜馬連、2番目にワイド〜三連単がある
                df = pd.concat([dfs[1], dfs[2]])

                df.index = [race_id] * len(df)
                return_tables[race_id] = df
            except IndexError:
                continue
            except AttributeError: #存在しないrace_idでAttributeErrorになるページもあるので追加
                continue
            except Exception as e:
                print(e)
                break
            except:
                break

        #pd.DataFrame型にして一つのデータにまとめる
        return_tables_df = pd.concat([return_tables[key] for key in return_tables])
        return return_tables_df

    @property
    def fukusho(self):
        fukusho = self.return_tables[self.return_tables[0]=='複勝'][[1,2]]
        wins = fukusho[1].str.split('br', expand=True)[[0,1,2]]

        wins.columns = ['win_0', 'win_1', 'win_2']
        returns = fukusho[2].str.split('br', expand=True)[[0,1,2]]
        returns.columns = ['return_0', 'return_1', 'return_2']

        df = pd.concat([wins, returns], axis=1)
        for column in df.columns:
            df[column] = df[column].str.replace(',', '')
        return df.fillna(0).astype(int)

    @property
    def tansho(self):
        tansho = self.return_tables[self.return_tables[0]=='単勝'][[1,2]]
        tansho.columns = ['win', 'return']

        for column in tansho.columns:
            tansho[column] = pd.to_numeric(tansho[column], errors='coerce')

        return tansho

    @property
    def umaren(self):
        umaren = self.return_tables[self.return_tables[0]=='馬連'][[1,2]]
        wins = umaren[1].str.split('-', expand=True)[[0,1]].add_prefix('win_')
        return_ = umaren[2].rename('return')
        df = pd.concat([wins, return_], axis=1)
        return df.apply(lambda x: pd.to_numeric(x, errors='coerce'))

    @property
    def umatan(self):
        umatan = self.return_tables[self.return_tables[0]=='馬単'][[1,2]]
        wins = umatan[1].str.split('→', expand=True)[[0,1]].add_prefix('win_')
        return_ = umatan[2].rename('return')
        df = pd.concat([wins, return_], axis=1)
        return df.apply(lambda x: pd.to_numeric(x, errors='coerce'))

    @property
    def wide(self):
        wide = self.return_tables[self.return_tables[0]=='ワイド'][[1,2]]
        wins = wide[1].str.split('br', expand=True)[[0,1,2]]
        wins = wins.stack().str.split('-', expand=True).add_prefix('win_')
        return_ = wide[2].str.split('br', expand=True)[[0,1,2]]
        return_ = return_.stack().rename('return')
        df = pd.concat([wins, return_], axis=1)
        return df.apply(lambda x: pd.to_numeric(x.str.replace(',',''), errors='coerce'))

    @property
    def sanrentan(self):
        rentan = self.return_tables[self.return_tables[0]=='三連単'][[1,2]]
        wins = rentan[1].str.split('→', expand=True)[[0,1,2]].add_prefix('win_')
        return_ = rentan[2].rename('return')
        df = pd.concat([wins, return_], axis=1)
        return df.apply(lambda x: pd.to_numeric(x, errors='coerce'))

    @property
    def sanrenpuku(self):
        renpuku = self.return_tables[self.return_tables[0]=='三連複'][[1,2]]
        wins = renpuku[1].str.split('-', expand=True)[[0,1,2]].add_prefix('win_')
        return_ = renpuku[2].rename('return')
        df = pd.concat([wins, return_], axis=1)
        return df.apply(lambda x: pd.to_numeric(x, errors='coerce'))

class ModelEvaluator:
    def __init__(self, model, return_tables_path_list):
        self.model = model
        self.rt = Return.read_pickle(return_tables_path_list)
        self.fukusho = self.rt.fukusho
        self.tansho = self.rt.tansho
        self.umaren = self.rt.umaren
        self.umatan = self.rt.umatan
        self.wide = self.rt.wide
        self.sanrentan = self.rt.sanrentan
        self.sanrenpuku = self.rt.sanrenpuku

    #3着以内に入る確率を予測
    def predict_proba(self, X, train=True, std=True, minmax=False):
        if train:
            proba = pd.Series(
                self.model.predict_proba(X.drop(['単勝'], axis=1))[:, 1], index=X.index
            )
        else:
            proba = pd.Series(
                self.model.predict_proba(X, axis=1)[:, 1], index=X.index
            )
        if std:
            #レース内で標準化して、相対評価する。「レース内偏差値」みたいなもの。
            standard_scaler = lambda x: (x - x.mean()) / x.std(ddof=0)
            proba = proba.groupby(level=0).transform(standard_scaler)
        if minmax:
            #データ全体を0~1にする
            proba = (proba - proba.min()) / (proba.max() - proba.min())
        return proba

    #0か1かを予測
    def predict(self, X, threshold=0.5):
        y_pred = self.predict_proba(X)
        self.proba = y_pred
        return [0 if p<threshold else 1 for p in y_pred]

    def score(self, y_true, X):
        return roc_auc_score(y_true, self.predict_proba(X))

    def feature_importance(self, X, n_display=20):
        importances = pd.DataFrame({"features": X.columns,
                                    "importance": self.model.feature_importances_})
        return importances.sort_values("importance", ascending=False)[:n_display]

    def pred_table(self, X, threshold=0.5, bet_only=True):
        pred_table = X.copy()[['馬番', '単勝']]
        pred_table['pred'] = self.predict(X, threshold)
        pred_table['score'] = self.proba
        if bet_only:
            return pred_table[pred_table['pred']==1][['馬番', '単勝', 'score']]
        else:
            return pred_table[['馬番', '単勝', 'score', 'pred']]

    def bet(self, race_id, kind, umaban, amount):
        if kind == 'fukusho':
            rt_1R = self.fukusho.loc[race_id]
            return_ = (rt_1R[['win_0', 'win_1', 'win_2']]==umaban).values * \
                rt_1R[['return_0', 'return_1', 'return_2']].values * amount/100
            return_ = np.sum(return_)
        if kind == 'tansho':
            rt_1R = self.tansho.loc[race_id]
            return_ = (rt_1R['win']==umaban) * rt_1R['return'] * amount/100
        if kind == 'umaren':
            rt_1R = self.umaren.loc[race_id]
            return_ = (set(rt_1R[['win_0', 'win_1']]) == set(umaban)) \
                * rt_1R['return']/100 * amount
        if kind == 'umatan':
            rt_1R = self.umatan.loc[race_id]
            return_ = (list(rt_1R[['win_0', 'win_1']]) == list(umaban))\
                * rt_1R['return']/100 * amount
        if kind == 'wide':
            rt_1R = self.wide.loc[race_id]
            return_ = (rt_1R[['win_0', 'win_1']].\
                           apply(lambda x: set(x)==set(umaban), axis=1)) \
                * rt_1R['return']/100 * amount
            return_ = return_.sum()
        if kind == 'sanrentan':
            rt_1R = self.sanrentan.loc[race_id]
            return_ = (list(rt_1R[['win_0', 'win_1', 'win_2']]) == list(umaban)) * \
                rt_1R['return']/100 * amount
        if kind == 'sanrenpuku':
            rt_1R = self.sanrenpuku.loc[race_id]
            return_ = (set(rt_1R[['win_0', 'win_1', 'win_2']]) == set(umaban)) \
                * rt_1R['return']/100 * amount
        if not (return_ >= 0):
                return_ = amount
        return return_

    def fukusho_return(self, X, threshold=0.5):
        pred_table = self.pred_table(X, threshold)
        n_bets = len(pred_table)

        return_list = []
        for race_id, preds in pred_table.groupby(level=0):
            return_list.append(np.sum([
                self.bet(race_id, 'fukusho', umaban, 1) for umaban in preds['馬番']
            ]))
        return_rate = np.sum(return_list) / n_bets
        std = np.std(return_list) * np.sqrt(len(return_list)) / n_bets
        n_hits = np.sum([x>0 for x in return_list])
        return n_bets, return_rate, n_hits, std

    def tansho_return(self, X, threshold=0.5):
        pred_table = self.pred_table(X, threshold)
        self.sample = pred_table
        n_bets = len(pred_table)

        return_list = []
        for race_id, preds in pred_table.groupby(level=0):
            return_list.append(
                np.sum([self.bet(race_id, 'tansho', umaban, 1) for umaban in preds['馬番']])
            )

        std = np.std(return_list) * np.sqrt(len(return_list)) / n_bets

        n_hits = np.sum([x>0 for x in return_list])
        return_rate = np.sum(return_list) / n_bets
        return n_bets, return_rate, n_hits, std

    def tansho_return_proper(self, X, threshold=0.5):
        pred_table = self.pred_table(X, threshold)
        n_bets = len(pred_table)

        return_list = []
        for race_id, preds in pred_table.groupby(level=0):
            return_list.append(
                np.sum(preds.apply(lambda x: self.bet(
                    race_id, 'tansho', x['馬番'], 1/x['単勝']), axis=1)))

        bet_money = (1 / pred_table['単勝']).sum()

        std = np.std(return_list) * np.sqrt(len(return_list)) / bet_money

        n_hits = np.sum([x>0 for x in return_list])
        return_rate = np.sum(return_list) / bet_money
        return n_bets, return_rate, n_hits, std

    def umaren_box(self, X, threshold=0.5, n_aite=5):
        pred_table = self.pred_table(X, threshold)
        n_bets = 0

        return_list = []
        for race_id, preds in pred_table.groupby(level=0):
            return_ = 0
            preds_jiku = preds.query('pred == 1')
            if len(preds_jiku) == 1:
                continue
            elif len(preds_jiku) >= 2:
                for umaban in combinations(preds_jiku['馬番'], 2):
                    return_ += self.bet(race_id, 'umaren', umaban, 1)
                    n_bets += 1
                return_list.append(return_)

        std = np.std(return_list) * np.sqrt(len(return_list)) / n_bets

        n_hits = np.sum([x>0 for x in return_list])
        return_rate = np.sum(return_list) / n_bets
        return n_bets, return_rate, n_hits, std

    def umatan_box(self, X, threshold=0.5, n_aite=5):
        pred_table = self.pred_table(X, threshold, bet_only = False)
        n_bets = 0

        return_list = []
        for race_id, preds in pred_table.groupby(level=0):
            return_ = 0
            preds_jiku = preds.query('pred == 1')
            if len(preds_jiku) == 1:
                continue
            elif len(preds_jiku) >= 2:
                for umaban in permutations(preds_jiku['馬番'], 2):
                    return_ += self.bet(race_id, 'umatan', umaban, 1)
                    n_bets += 1
            return_list.append(return_)

        std = np.std(return_list) * np.sqrt(len(return_list)) / n_bets

        n_hits = np.sum([x>0 for x in return_list])
        return_rate = np.sum(return_list) / n_bets
        return n_bets, return_rate, n_hits, std

    def wide_box(self, X, threshold=0.5, n_aite=5):
        pred_table = self.pred_table(X, threshold, bet_only = False)
        n_bets = 0

        return_list = []
        for race_id, preds in pred_table.groupby(level=0):
            return_ = 0
            preds_jiku = preds.query('pred == 1')
            if len(preds_jiku) == 1:
                continue
            elif len(preds_jiku) >= 2:
                for umaban in combinations(preds_jiku['馬番'], 2):
                    return_ += self.bet(race_id, 'wide', umaban, 1)
                    n_bets += 1
                return_list.append(return_)

        std = np.std(return_list) * np.sqrt(len(return_list)) / n_bets

        n_hits = np.sum([x>0 for x in return_list])
        return_rate = np.sum(return_list) / n_bets
        return n_bets, return_rate, n_hits, std

    def sanrentan_box(self, X, threshold=0.5):
        pred_table = self.pred_table(X, threshold)
        n_bets = 0

        return_list = []
        for race_id, preds in pred_table.groupby(level=0):
            return_ = 0
            if len(preds)<3:
                continue
            else:
                for umaban in permutations(preds['馬番'], 3):
                    return_ += self.bet(race_id, 'sanrentan', umaban, 1)
                    n_bets += 1
                return_list.append(return_)

        std = np.std(return_list) * np.sqrt(len(return_list)) / n_bets

        n_hits = np.sum([x>0 for x in return_list])
        return_rate = np.sum(return_list) / n_bets
        return n_bets, return_rate, n_hits, std

    def sanrenpuku_box(self, X, threshold=0.5):
        pred_table = self.pred_table(X, threshold)
        n_bets = 0

        return_list = []
        for race_id, preds in pred_table.groupby(level=0):
            return_ = 0
            if len(preds)<3:
                continue
            else:
                for umaban in combinations(preds['馬番'], 3):
                    return_ += self.bet(race_id, 'sanrenpuku', umaban, 1)
                    n_bets += 1
                return_list.append(return_)

        std = np.std(return_list) * np.sqrt(len(return_list)) / n_bets

        n_hits = np.sum([x>0 for x in return_list])
        return_rate = np.sum(return_list) / n_bets
        return n_bets, return_rate, n_hits, std

    def umaren_nagashi(self, X, threshold=0.5, n_aite=5):
        pred_table = self.pred_table(X, threshold, bet_only = False)
        n_bets = 0

        return_list = []
        for race_id, preds in pred_table.groupby(level=0):
            return_ = 0
            preds_jiku = preds.query('pred == 1')
            if len(preds_jiku) == 1:
                preds_aite = preds.sort_values('score', ascending = False)\
                    .iloc[1:(n_aite+1)]['馬番']
                return_ = preds_aite.map(
                    lambda x: self.bet(
                        race_id, 'umaren', [preds_jiku['馬番'].values[0], x], 1
                    )
                ).sum()
                n_bets += n_aite
                return_list.append(return_)
            elif len(preds_jiku) >= 2:
                for umaban in combinations(preds_jiku['馬番'], 2):
                    return_ += self.bet(race_id, 'umaren', umaban, 1)
                    n_bets += 1
                return_list.append(return_)

        std = np.std(return_list) * np.sqrt(len(return_list)) / n_bets

        n_hits = np.sum([x>0 for x in return_list])
        return_rate = np.sum(return_list) / n_bets
        return n_bets, return_rate, n_hits, std

    def umatan_nagashi(self, X, threshold=0.5, n_aite=5):
        pred_table = self.pred_table(X, threshold, bet_only = False)
        n_bets = 0

        return_list = []
        for race_id, preds in pred_table.groupby(level=0):
            return_ = 0
            preds_jiku = preds.query('pred == 1')
            if len(preds_jiku) == 1:
                preds_aite = preds.sort_values('score', ascending = False)\
                    .iloc[1:(n_aite+1)]['馬番']
                return_ = preds_aite.map(
                    lambda x: self.bet(
                        race_id, 'umatan', [preds_jiku['馬番'].values[0], x], 1
                    )
                ).sum()
                n_bets += n_aite

            elif len(preds_jiku) >= 2:
                for umaban in permutations(preds_jiku['馬番'], 2):
                    return_ += self.bet(race_id, 'umatan', umaban, 1)
                    n_bets += 1
            return_list.append(return_)

        std = np.std(return_list) * np.sqrt(len(return_list)) / n_bets

        n_hits = np.sum([x>0 for x in return_list])
        return_rate = np.sum(return_list) / n_bets
        return n_bets, return_rate, n_hits, std

    def wide_nagashi(self, X, threshold=0.5, n_aite=5):
        pred_table = self.pred_table(X, threshold, bet_only = False)
        n_bets = 0

        return_list = []
        for race_id, preds in pred_table.groupby(level=0):
            return_ = 0
            preds_jiku = preds.query('pred == 1')
            if len(preds_jiku) == 1:
                preds_aite = preds.sort_values('score', ascending = False)\
                    .iloc[1:(n_aite+1)]['馬番']
                return_ = preds_aite.map(
                    lambda x: self.bet(
                        race_id, 'wide', [preds_jiku['馬番'].values[0], x], 1
                    )
                ).sum()
                n_bets += len(preds_aite)
                return_list.append(return_)
            elif len(preds_jiku) >= 2:
                for umaban in combinations(preds_jiku['馬番'], 2):
                    return_ += self.bet(race_id, 'wide', umaban, 1)
                    n_bets += 1
                return_list.append(return_)

        std = np.std(return_list) * np.sqrt(len(return_list)) / n_bets

        n_hits = np.sum([x>0 for x in return_list])
        return_rate = np.sum(return_list) / n_bets
        return n_bets, return_rate, n_hits, std

    def sanrentan_nagashi(self, X, threshold = 1.5, n_aite=7):
        pred_table = self.pred_table(X, threshold, bet_only = False)
        n_bets = 0
        return_list = []
        for race_id, preds in pred_table.groupby(level=0):
            preds_jiku = preds.query('pred == 1')
            if len(preds_jiku) == 1:
                continue
            elif len(preds_jiku) == 2:
                preds_aite = preds.sort_values('score', ascending = False)\
                    .iloc[2:(n_aite+2)]['馬番']
                return_ = preds_aite.map(
                    lambda x: self.bet(
                        race_id, 'sanrentan',
                        np.append(preds_jiku['馬番'].values, x),
                        1
                    )
                ).sum()
                n_bets += len(preds_aite)
                return_list.append(return_)
            elif len(preds_jiku) >= 3:
                return_ = 0
                for umaban in permutations(preds_jiku['馬番'], 3):
                    return_ += self.bet(race_id, 'sanrentan', umaban, 1)
                    n_bets += 1
                return_list.append(return_)

        std = np.std(return_list) * np.sqrt(len(return_list)) / n_bets

        n_hits = np.sum([x>0 for x in return_list])
        return_rate = np.sum(return_list) / n_bets
        return n_bets, return_rate, n_hits, std





# データをロードする関数
def load_data(race_id):
    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    html = requests.get(url)
    html.encoding = "EUC-JP"

    try:
        df = pd.read_html(StringIO(html.text))[0]
        df.columns = df.iloc[0]
        column_names = ['枠', '馬 番', '印', '馬名', '性齢', '斤量', '騎手', '厩舎', '馬体重 (増減)', '予想オッズ', '人気', '登録', 'メモ']
        df.columns = column_names
        df.drop(['印', '登録', 'メモ', '予想オッズ', '人気'], axis=1, inplace=True)

        # 新しい列 'AI予想' を先頭に追加
        df.insert(0, 'AI予想', None)  # 0は挿入位置、'AI予想'は列名、Noneは初期値
        df['race_id'] = race_id
        return df
    except Exception as e:
        st.write(f"エラーが発生しました: {e}")
        return None

from bs4 import BeautifulSoup

def load_additional_data(race_id):
    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
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


# Streamlit UI
st.title("競馬AI予想🐎")
day_options = ["1日目", "2日目", "3日目"]
selected_day = st.selectbox("何日目か選択してください", day_options)
day_number = int(selected_day[0])

racecourse_map = {
    "札幌": "01",
    "函館": "02",
    "福島": "03",
    "新潟": "04",
    "東京": "05",
    "中山": "06",
    "中京": "07",
    "京都": "08",
    "阪神": "09",
    "小倉": "10"
}

racecourse = st.selectbox("競馬場を選択してください", list(racecourse_map.keys()))
holding_number = st.selectbox("開催回数を選択してください", list(range(1, 12)))

base_race_id = f"2023{racecourse_map[racecourse]}{holding_number:02d}{day_number:02d}"
st.write(f"生成されたbase_race_idは {base_race_id} です。")
st.write(f"サンプルは「第1回札幌競馬場1日目1レース」の3位以内に入る確率を、機械学習にて予測しております。")



sample = pd.read_pickle('sample.pickle')





#時系列に沿って訓練データとテストデータに分ける関数
def split_data(df, test_size=0.3):
    sorted_id_list = df.sort_values("date").index.unique()
    train_id_list = sorted_id_list[: round(len(sorted_id_list) * (1 - test_size))]
    test_id_list = sorted_id_list[round(len(sorted_id_list) * (1 - test_size)) :]
    train = df.loc[train_id_list]
    test = df.loc[test_id_list]
    return train, test

sample = sample.drop(['rank', 'date', '単勝'], axis=1)

# 追加したい列名のリスト
columns_to_add = ['weather_雪', 'weather_小雪', 'ground_state_重', 'weather_小雨', 'weather_曇', 'weather_雨', 'race_type_障害', 'ground_state_稍重', 'ground_state_不良']

# データフレームに存在しない列名だけを追加
for col in columns_to_add:
    if col not in sample.columns:
        sample[col] = 0  # 数値の0を入れる

# LightGBMモデルを読み込む
lgb_clf = lgb.Booster(model_file="lgb_model.txt")

# 予測を実施
predictions = lgb_clf.predict(sample)

# 予測結果をdata_cに追加
sample['Predicted_Rank'] = predictions

# 予測結果を降順にソート
sorted_predictions = sample.sort_values(by=['Predicted_Rank'], ascending=False)

# TOP3の予測結果を抽出
top_3_per_race = sample.groupby(level=0).apply(lambda x: x.nlargest(3, 'Predicted_Rank'))

# 各レースごとに出馬表とTOP3を表示
for race_id, group_data in sorted_predictions.groupby(level=0):

    # df（出馬表）を対応するrace_idで更新する
    df = load_data(race_id)

    if df is not None:
        # race_idと馬番が一致する行の'AI予想'列にPredicted_Rankの値を挿入
        for _, row in group_data.iterrows():
            horse_number = row['馬番']
            predicted_rank = row['Predicted_Rank']
            df.loc[df['馬 番'] == horse_number, 'AI予想'] = predicted_rank

        # 追加の情報とともに出馬表を表示
        additional_data = load_additional_data(race_id)
        if additional_data:
            st.markdown(f"**サンプル:** {additional_data['race_name']}")


        st.dataframe(df)

    # そのレースのTOP3予測を表示
    if race_id in top_3_per_race.index.levels[0]:
        top_3_data = top_3_per_race.loc[race_id]
        st.dataframe(top_3_data[['馬番', 'Predicted_Rank']])
    break



if st.button('AI予想'):
    st.write('AI予想を開始致します。処理には15分〜20分かかります。')
    

    # race_id_list の生成
    race_id_list = [f"{base_race_id}{str(i).zfill(2)}" for i in range(1, 13)]


    # Results クラスの scrape メソッドを使用
    results = Results_1.scrape(race_id_list)
    results = results.rename(columns=lambda x: x.replace(' ', ''))
    st.write("出馬表: ", results)
    race_id_list = results.index.unique()

    #return_tables = Return.scrape(race_id_list)
    horse_id_list = results['horse_id'].unique()
    horse_results = HorseResults_1.scrape(horse_id_list)
    horse_results = horse_results.rename(columns=lambda x: x.replace(' ', ''))
    st.write("出走馬の過去成績情報: ", horse_results)

    peds = Peds.scrape(horse_id_list)
    st.write("出走馬の血統情報: ", peds)

    #Resultsクラスのオブジェクト作成
    r = Results_2(results)
    #前処理
    r.preprocessing()

    #馬の過去成績データ追加
    hr = HorseResults_2(horse_results)
    r.merge_horse_results(hr)

    #血統データ追加
    p = Peds(peds)
    p.encode()
    r.merge_horse_results(hr, n_samples_list=[5, 9, 'all'])
    r.merge_peds(p.peds_e)
    r.process_categorical()


    # LightGBMモデルを読み込む
    #lgb_clf = lgb.Booster(model_file="/Users/yamanekousaku/Desktop/競馬予想AI/機械学習モデル/lgb_model.txt")

    # 以下は既存のStreamlitアプリのコード
    # ...

    #時系列に沿って訓練データとテストデータに分ける関数
    def split_data(df, test_size=0.3):
        sorted_id_list = df.sort_values("date").index.unique()
        train_id_list = sorted_id_list[: round(len(sorted_id_list) * (1 - test_size))]
        test_id_list = sorted_id_list[round(len(sorted_id_list) * (1 - test_size)) :]
        train = df.loc[train_id_list]
        test = df.loc[test_id_list]
        return train, test

    data_c =  r.data_c.drop(['rank', 'date', '単勝'], axis=1)

    # 追加したい列名のリスト
    columns_to_add = ['weather_雪', 'weather_小雪', 'ground_state_重', 'weather_小雨', 'weather_曇', 'weather_雨', 'race_type_障害', 'ground_state_稍重', 'ground_state_不良']

    # データフレームに存在しない列名だけを追加
    for col in columns_to_add:
        if col not in data_c.columns:
            data_c[col] = 0  # 数値の0を入れる

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