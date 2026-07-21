import pandas as pd
import requests_cache 
from retry_requests import retry
import openmeteo_requests
import os
import matplotlib.pyplot as plt
import numpy as np
import holidays
from sklearn.linear_model import LinearRegression 
import xgboost
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import sys

###Classes and Functions

class DemandDataset(Dataset):
    def __init__(self, seq, stat, tgt, window=7):
        self.seq, self.stat, self.tgt, self.w = seq, stat, tgt, window
    def __len__(self):
        return len(self.tgt) - self.w
    def __getitem__(self, i):
        d = i + self.w
        return (torch.tensor(self.seq[d-self.w:d].reshape(-1, 2)),
                torch.tensor(self.stat[d]),
                torch.tensor(self.tgt[d]))

class ForecastModelv0(nn.Module):
    def __init__(self, seqchannels: int, n: int, hiddenunits: int, outputshape: int = 48):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size = seqchannels,
            hidden_size = hiddenunits,
            batch_first = True)

        self.head = nn.Sequential(
            nn.Linear(in_features = hiddenunits + n, out_features= 96),
            nn.ReLU(),
            nn.Linear(in_features=96, out_features=outputshape)
        )

    def forward(self, seqinput, staticinput):
        step_outputs, (finalhidden, finalcell) = self.lstm(seqinput)
        weeksummary = finalhidden[-1]
        combined = torch.cat([weeksummary, staticinput], dim = 1)
        return self.head(combined)

def parse_dates(s):
    for fmt in FORMATS:
        try:
            return pd.to_datetime(s, format=fmt)
        except ValueError:
            continue
    raise ValueError(f"No known format matches: {s.iloc[0]!r}")

###Paths

filedir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
demanddir = os.path.join(filedir, "Data - National Demand")
weathdir = os.path.join(filedir, "Data - Weather")
resultsdir = os.path.join(filedir, "Results")
weathfile = os.path.join(weathdir, "weather.csv")
os.makedirs(demanddir, exist_ok=True)
os.makedirs(weathdir, exist_ok=True)
os.makedirs(resultsdir, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

###Collect electricity demand data

dfs = []
years = []

while True:
    demand_choice = input("Do demanddata_YEAR.csv files exist in the Data - National Demand directory? [y/n]: ").strip().lower()
    if demand_choice in ("y", "yes"):
        break
    elif demand_choice in ("n", "no"):
        print("Please add the files to the directory and restart the file.")
        sys.exit(0)
    else:
        print("Please answer y/n.")

for year in range(2016, 2027):
    df = pd.read_csv(os.path.join(demanddir, f"demanddata_{year}.csv"))
    dfs.append(df)
    years.append(year) #Create list of frames

keeps = ["SETTLEMENT_DATE","SETTLEMENT_PERIOD", "FORECAST_ACTUAL_INDICATOR","ND"]

for i, df in enumerate(dfs):
    dfs[i] = df[[col for col in keeps if col in df.columns]]


for i, df in enumerate(dfs):
    if "FORECAST_ACTUAL_INDICATOR" in df.columns:
        dfs[i] = df[df["FORECAST_ACTUAL_INDICATOR"] != "F"].drop(columns="FORECAST_ACTUAL_INDICATOR") # no forecasts, column can be dropped

#Add dates for working
FORMATS = ["%Y-%m-%d", "%d-%b-%Y", "%d-%b-%y", "%d/%m/%Y"]

for df in dfs:
    df["SETTLEMENT_DATE"] = parse_dates(df["SETTLEMENT_DATE"])
    dfindexer = df["SETTLEMENT_DATE"] + pd.to_timedelta((df["SETTLEMENT_PERIOD"] - 1) * 30, unit="m") #index
    position = df.columns.get_loc("SETTLEMENT_DATE")
    df.insert(position, "YEAR", df["SETTLEMENT_DATE"].dt.year)
    df.insert(position+1, "MONTH", df["SETTLEMENT_DATE"].dt.month)
    df.insert(position+2, "DAY", df["SETTLEMENT_DATE"].dt.day)
    df.insert(position+3, "DoW", df["SETTLEMENT_DATE"].dt.dayofweek)
    df.insert(position+4, "HOUR", (df["SETTLEMENT_PERIOD"] - 1) // 2)
    df.insert(position+5, "DoY", df["SETTLEMENT_DATE"].dt.day_of_year)
    df.index = dfindexer

#Add holidays
hols=holidays.country_holidays("GB", subdiv = "ENG", years = range(2016,2027))
for year, df in zip(years, dfs):
    df["HOLIDAY"] = df["SETTLEMENT_DATE"].dt.date.isin(hols)






###Collect weather data
# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below

while True:
    weather_choice = input("Do you wish to download weather data to the Data - Weather directory?\n If data currently exists, this will be overwritten.\n [y/n]: ").strip().lower()
    if weather_choice in ("y", "yes"):
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": 52.68,
            "longitude": 1.54,
            "start_date": "2016-01-01",
            "end_date": "2026-07-15",
            "hourly": ["apparent_temperature", "cloud_cover"],
        }
        responses = openmeteo.weather_api(url, params = params)

        # Process first location. Add a for-loop for multiple locations or weather models
        response = responses[0]

        # Process hourly data. The order of variables needs to be the same as requested.
        hourly = response.Hourly()
        hourly_apparent_temperature = hourly.Variables(0).ValuesAsNumpy()
        hourly_cloud_cover = hourly.Variables(1).ValuesAsNumpy()

        hourly_data = {
            "date": pd.date_range(
                start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
                end =  pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
                freq = pd.Timedelta(seconds = hourly.Interval()),
                inclusive = "left"
            )
        }

        hourly_data["apparent_temperature"] = hourly_apparent_temperature
        hourly_data["cloud_cover"] = hourly_cloud_cover

        hourly_dataframe = pd.DataFrame(data = hourly_data)
        hourly_dataframe.to_csv(weathfile, index = False)
        break

    elif weather_choice in ("n", "no"):
        break
    else:
        print("Please answer y/n.")

###Process weather data
weather = pd.read_csv(weathfile)

#Infer data for 30 mins and interpolation
weather["date"] = pd.to_datetime(weather["date"], utc=True)
weather = weather.set_index("date", drop=False).sort_index()
weather = weather.drop(columns = "date")
ind = pd.date_range(weather.index.min(), weather.index.max() + pd.Timedelta(minutes = 30), freq = "30min", tz = weather.index.tz)
weather = weather.reindex(ind)
weather[["apparent_temperature", "cloud_cover"]] = (weather[["apparent_temperature", "cloud_cover"]].interpolate(method = "time", limit = 2)) #Interpolation

#Add same columns as df
local = weather.index.tz_convert("Europe/London")
weather["SETTLEMENT_DATE"] = local.normalize().tz_localize(None)
weather["SETTLEMENT_PERIOD"] = (local.hour * 60 + local.minute) // 30 + 1 


###Merge data
demanddata = pd.concat(dfs)
demanddata["SETTLEMENT_DATE"] = demanddata["SETTLEMENT_DATE"].dt.tz_localize(None).dt.normalize()
weather["SETTLEMENT_DATE"] = weather["SETTLEMENT_DATE"].dt.tz_localize(None)
weather = weather[["SETTLEMENT_DATE", "SETTLEMENT_PERIOD", "apparent_temperature", "cloud_cover"]]
demanddata = demanddata.merge(weather, on = ["SETTLEMENT_PERIOD", "SETTLEMENT_DATE"], how = "left")



###Baseline Naive Model

baseline = demanddata
baseline["predyesterday"] = baseline.groupby("SETTLEMENT_PERIOD")["ND"].shift(1) #1 day prediction
baseline["predlastweek"]  = baseline.groupby("SETTLEMENT_PERIOD")["ND"].shift(7) #1 week prediction

evaldat = baseline[baseline["SETTLEMENT_DATE"].dt.year == 2025].dropna(subset=["predlastweek"])

baseweekper = (abs(evaldat["ND"] - evaldat["predlastweek"]) / evaldat["ND"]).mean() * 100
baseweekrms = ((evaldat["ND"] - evaldat["predlastweek"])**2).mean()**0.5
basedayper = (abs(evaldat["ND"] - evaldat["predyesterday"]) / evaldat["ND"]).mean() * 100
basedayrms = ((evaldat["ND"] - evaldat["predyesterday"])**2).mean()**0.5

###Feature Engineering

#Cycles and weekends
demanddata["WEEKEND"] =(demanddata["DoW"] >=5).astype(int)
demanddata["SIN_PERIOD"] = np.sin(2*np.pi*(demanddata["SETTLEMENT_PERIOD"]-1)/48)
demanddata["COS_PERIOD"] = np.cos(2*np.pi*(demanddata["SETTLEMENT_PERIOD"]-1)/48)
demanddata["SIN_DOY"] = np.sin(2*np.pi*(demanddata["DoY"])/365.25)
demanddata["COS_DOY"] = np.cos(2*np.pi*(demanddata["DoY"])/365.25)

#Heating turned on at 15C
demanddata["hdd"] = (15 - demanddata["apparent_temperature"]).clip(lower=0)   # heating degree

#Yesterdays demand 
lk = demanddata[["SETTLEMENT_DATE", "SETTLEMENT_PERIOD", "ND"]].copy()
for d, name in [(1, "nd_lag1d"), (7, "nd_lag7d")]:
    tmp = lk.copy()
    tmp["SETTLEMENT_DATE"] = tmp["SETTLEMENT_DATE"] + pd.Timedelta(days=d)
    demanddata = demanddata.merge(tmp.rename(columns={"ND": name}),
                  on=["SETTLEMENT_DATE", "SETTLEMENT_PERIOD"], how="left")
daymean = demanddata.groupby("SETTLEMENT_DATE")["ND"].mean().rename("nd_prevdaymean")
daymean.index = daymean.index + pd.Timedelta(days=1)
demanddata = demanddata.merge(daymean, left_on="SETTLEMENT_DATE", right_index=True, how="left")

#Remove na (present due to discontinuous data guaranteed at start)
demanddata = demanddata.dropna(subset=["nd_lag7d", "hdd"])


###Linear Regression Model

lindata = demanddata.drop(columns = ["HOUR", "DAY","MONTH","YEAR","SETTLEMENT_PERIOD","predyesterday","predlastweek"])
xcols = ["SIN_PERIOD", "COS_PERIOD", "SIN_DOY", "COS_DOY", "WEEKEND", "HOLIDAY", "hdd", "apparent_temperature", "cloud_cover", "nd_lag1d", "nd_lag7d", "nd_prevdaymean"]
ycols = "ND"

lindata = lindata.dropna(subset=xcols) #Remove na in lag data from missing NESO data

train = lindata[lindata["SETTLEMENT_DATE"] <"2024-01-01"]
test = lindata[lindata["SETTLEMENT_DATE"] >="2025-01-01" & (df["SETTLEMENT_DATE"] < "2026-01-01")]

linmod = LinearRegression().fit(train[xcols], train[ycols])
predictions = linmod.predict(test[xcols])

linerperc = (abs(test[ycols] - predictions) / test[ycols]).mean()*100
linrms = ((test[ycols] - predictions)**2).mean()**0.5


###Grad Boosted Model

graddata = demanddata.drop(columns = ["HOUR", "DAY","MONTH","YEAR","predyesterday","predlastweek"])

y = ycols
X = xcols + ["DoW"] + ["SETTLEMENT_PERIOD"]

train = graddata[graddata["SETTLEMENT_DATE"] < "2024-01-01"]
val   = graddata[(graddata["SETTLEMENT_DATE"] >= "2024-01-01") & (graddata["SETTLEMENT_DATE"] < "2025-01-01")]
test  = graddata[graddata["SETTLEMENT_DATE"] >= "2025-01-01" & (df["SETTLEMENT_DATE"] < "2026-01-01")]

gradboomod = xgboost.XGBRegressor(n_estimators = 1000,
                                max_depth = 6,
                                learning_rate = 0.05,
                                early_stopping_rounds = 70,
                                eval_metric = 'rmse',
                                random_state = 42)


gradboomod.fit(train[X], train[y], eval_set = [(val[X], val[y])], verbose = 80)

y_preds = gradboomod.predict(test[X])
gbmperc = (abs(test[y] - y_preds) / test[y]).mean()*100
gbmrms = ((test[y] - y_preds)**2).mean()**0.5

#Record Results
xgboosttab = test.copy()
xgboosttab["preds"]  = y_preds
xgboosttab["Error%"] = abs(xgboosttab["ND"] - xgboosttab["preds"]) / xgboosttab["ND"] * 100

by_month  = xgboosttab.groupby(xgboosttab["SETTLEMENT_DATE"].dt.month)["Error%"].mean()
by_dow    = xgboosttab.groupby("DoW")["Error%"].mean()
by_period = xgboosttab.groupby("SETTLEMENT_PERIOD")["Error%"].mean()
worst_days = (xgboosttab.groupby("SETTLEMENT_DATE")["Error%"].mean()
                 .sort_values(ascending=False).head(10))

context = xgboosttab.groupby("SETTLEMENT_DATE").agg(ape=("Error%","mean"),
                                             hol=("HOLIDAY","max"),
                                             temp=("apparent_temperature","mean"))

#Rolling month test
months = pd.date_range("2025-01-01", "2026-06-01", freq="MS")
rolled = []

for m in months:
                train = graddata[graddata["SETTLEMENT_DATE"] < m]
                test  = graddata[(graddata["SETTLEMENT_DATE"] >= m) & (graddata["SETTLEMENT_DATE"] < m + pd.offsets.MonthBegin(1))]

                gradroll = xgboost.XGBRegressor(n_estimators = 500,
                                                max_depth = 6,
                                                learning_rate = 0.05,
                                                eval_metric = 'rmse',
                                                random_state = 42)


                gradroll.fit(train[X], train[y], verbose = False)

                ps = gradroll.predict(test[X])
                rollperc = (abs(test["ND"] - ps) / test["ND"]).mean()*100
                rollmrms = ((test["ND"] - ps)**2).mean()**0.5
                rolled.append({"month": m, "Error%": rollperc, "RMS": rollmrms})
rolltable = pd.DataFrame(rolled).set_index("month")

###RNN Model
nndata = demanddata.drop(columns = ["YEAR", "MONTH", "DAY", "HOUR", "DoY", "cloud_cover", "predyesterday", "predlastweek", "SIN_PERIOD", "COS_PERIOD", "nd_lag1d", "nd_lag7d", "nd_prevdaymean"]).copy()

#normalise based on training data

train_normalise_filter = nndata["SETTLEMENT_DATE"] < "2024-01-01"

NDmean, NDstd = nndata.loc[train_normalise_filter, "ND"].mean(), nndata.loc[train_normalise_filter, "ND"].std()
Tmean, Tstd = nndata.loc[train_normalise_filter, "apparent_temperature"].mean(), nndata.loc[train_normalise_filter, "apparent_temperature"].std()
HDDmean, HDDstd = nndata.loc[train_normalise_filter, "hdd"].mean(), nndata.loc[train_normalise_filter, "hdd"].std()

nndata["NDs"] = (nndata["ND"] - NDmean) / NDstd
nndata["TEMPs"] = (nndata["apparent_temperature"] - Tmean) / Tstd
nndata["HDDs"] = (nndata["hdd"] - HDDmean) / HDDstd
nndata["HOLIDAY"] = nndata["HOLIDAY"].astype(float)
nndata = pd.get_dummies(nndata, columns=["DoW"], prefix="dow", dtype=float) 

BATCH_SIZE = 32

nd_w  = nndata.pivot_table(index="SETTLEMENT_DATE", columns="SETTLEMENT_PERIOD", values="NDs")
tmp_w = nndata.pivot_table(index="SETTLEMENT_DATE", columns="SETTLEMENT_PERIOD", values="TEMPs")

good = nd_w.notna().all(axis=1) & tmp_w.notna().all(axis=1)
nd_w, tmp_w = nd_w[good], tmp_w[good]
days = nd_w.index

statcols = ["SIN_DOY", "COS_DOY", "WEEKEND", "HOLIDAY", "HDDs", "TEMPs"] + \
           [c for c in nndata.columns if c.startswith("dow_")]
stat = nndata.groupby("SETTLEMENT_DATE")[statcols].mean().loc[days]

seq_arr  = np.stack([nd_w.values, tmp_w.values], axis=-1).astype("float32")  # (n_days, 48, 2)
stat_arr = stat.values.astype("float32")                                      # (n_days, 13)
tgt_arr  = nd_w.values.astype("float32")                                      # (n_days, 48)
    

tr_end = days.searchsorted(pd.Timestamp("2024-01-01"))
va_end = days.searchsorted(pd.Timestamp("2025-01-01"))
te_end = days.searchsorted(pd.Timestamp("2026-01-01"))

train_data = DemandDataset(seq_arr[:tr_end],         stat_arr[:tr_end],         tgt_arr[:tr_end])
val_data   = DemandDataset(seq_arr[tr_end-7:va_end], stat_arr[tr_end-7:va_end], tgt_arr[tr_end-7:va_end])
test_data  = DemandDataset(seq_arr[va_end-7:te_end], stat_arr[va_end-7:te_end], tgt_arr[va_end-7:te_end])    

#Turn datasets into iterables
train_dataloader = DataLoader(train_data,
                              batch_size = BATCH_SIZE,
                              shuffle = True
)

val_dataloader = DataLoader(val_data,
                              batch_size = BATCH_SIZE,
                              shuffle = False
)

test_dataloader = DataLoader(test_data,
                             batch_size = BATCH_SIZE,
                             shuffle = False
                             )

#Check this
print(f"Dataloaders: {train_dataloader, val_dataloader, test_dataloader}")
print(f"Length of train dataloader: {len(train_dataloader)} batches of {BATCH_SIZE}")
print(f"Length of val dataloader: {len(val_dataloader)} batches of {BATCH_SIZE}")
print(f"Length of test dataloader: {len(test_dataloader)} batches of size {BATCH_SIZE}")

#RNN

results = []

for seed in range(40,50):
    torch.manual_seed(seed)

    model_0 = ForecastModelv0(seqchannels=2,
                            n = 13,
                            hiddenunits=96,
                            outputshape = 48)
    model_0.to(device)

    loss_fn = nn.MSELoss()
    optimiser = torch.optim.Adam(model_0.parameters(), lr = 0.001)

    epochs = 100
    best_val = float("inf")
    patience = 10
    bad = 0
    #T and T loop

    for epoch in range(epochs):
        #TRAINING
        model_0.train()
        train_loss=0
        #Add a loop to loop through training batches
        for (X_seq, X_stat, y) in train_dataloader:
            X_seq, X_stat, y = X_seq.to(device), X_stat.to(device), y.to(device)
            #1. FP
            y_pred = model_0(X_seq, X_stat)
            #2. Calc loss per Batch
            loss = loss_fn(y_pred, y)
            train_loss += loss.item() # accumulatively add up the loss per epoch
            #3. Optimiser zero grad
            optimiser.zero_grad()
            #4. Loss backward
            loss.backward()
            #5. Optimiser step
            optimiser.step()
        
        train_loss /= len(train_dataloader)

        #VALIDATION
        model_0.eval()
        val_loss = 0
        with torch.inference_mode():
            for X_seq, X_stat, y in val_dataloader:
                X_seq, X_stat, y = X_seq.to(device), X_stat.to(device), y.to(device)
                #1. FP
                val_pred = model_0(X_seq, X_stat)
                #2. Loss
                val_loss += loss_fn(val_pred, y).item()
        val_loss /= len(val_dataloader)

        if epoch % 10 == 0:
            print(f"Epoch: {epoch:3d} Train loss: {train_loss:.5f} Val loss: {val_loss:.5f}")

        if val_loss < best_val - 0.00001:
            best_val, bad = val_loss, 0
            torch.save(model_0.state_dict(), "best.pt")
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stop at epoch {epoch}, best validation loss at {best_val:.5f}")
                break

    model_0.load_state_dict(torch.load("best.pt", weights_only= True))

    #TESTING

    pred, actual = [], []
    model_0.eval()
    with torch.inference_mode():
        for X_seq, X_stat, y in test_dataloader:
            X_seq, X_stat, y = X_seq.to(device), X_stat.to(device), y.to(device)
            #1. Forward pass
            pred.append(model_0(X_seq, X_stat))
            actual.append(y)

    pred   = torch.cat(pred).to("cpu").numpy()   * NDstd + NDmean
    actual = torch.cat(actual).to("cpu").numpy() * NDstd + NDmean

    lstmperc = (abs(actual - pred) / actual).mean()*100
    lstmrmse = ((actual - pred)**2).mean()**0.5
    results.append({"seed":seed, "Error %": lstmperc, "RMS": lstmrmse})

### Output
resultsframe = pd.DataFrame(results)
lstm_erroravg = resultsframe["Error %"].mean()
lstm_rmsavg = resultsframe["RMS"].mean()

summary = pd.DataFrame([
    {"Model": "Baseline (day)",           "Error %": basedayper,    "RMS": basedayrms},
    {"Model": "Baseline (week)",          "Error %": baseweekper,   "RMS": baseweekrms},
    {"Model": "Linear Regression",        "Error %": linerperc,     "RMS": linrms},
    {"Model": "XGBoost",                  "Error %": gbmperc,       "RMS": gbmrms},
    {"Model": "LSTM (avg, seeds 40-49)",  "Error %": lstm_erroravg, "RMS": lstm_rmsavg},
])

summary.to_csv(os.path.join(resultsdir, "model_summary.csv"), index=False)
resultsframe.to_csv(os.path.join(resultsdir, "lstm_seed_results.csv"), index=False)
rolltable.to_csv(os.path.join(resultsdir, "rolling_monthly_results.csv"))