import xarray as xr
import numpy as np

ds = xr.open_dataset('data/processed/era5_hanoi.nc')

time_dim = 'valid_time'

print(f"Temps : {ds[time_dim].values[:6]}...")
print(f"Latitude  : {ds.latitude.values}")
print(f"Longitude : {ds.longitude.values}")

lat_humg = 21.0
lon_humg = 105.8

blh = ds['blh'].sel(latitude=lat_humg, longitude=lon_humg, method='nearest')
u10 = ds['u10'].sel(latitude=lat_humg, longitude=lon_humg, method='nearest')
v10 = ds['v10'].sel(latitude=lat_humg, longitude=lon_humg, method='nearest')
wind_speed = np.sqrt(u10**2 + v10**2)

print("\n=== PBLH PAR HEURE ===")
for hour in [8, 14, 22]:
    blh_h  = blh.where(blh[time_dim].dt.hour == hour, drop=True)
    ws_h   = wind_speed.where(wind_speed[time_dim].dt.hour == hour, drop=True)

    blh_jan = blh_h.where(blh_h[time_dim].dt.month == 1, drop=True)
    blh_jul = blh_h.where(blh_h[time_dim].dt.month == 7, drop=True)
    ws_jan  = ws_h.where(ws_h[time_dim].dt.month == 1, drop=True)
    ws_jul  = ws_h.where(ws_h[time_dim].dt.month == 7, drop=True)

    print(f"\n  {hour:02d}:00")
    print(f"    PBLH janvier : {float(blh_jan.mean()):6.0f} m   (std: {float(blh_jan.std()):4.0f} m)")
    print(f"    PBLH juillet : {float(blh_jul.mean()):6.0f} m   (std: {float(blh_jul.std()):4.0f} m)")
    print(f"    Vent jan     : {float(ws_jan.mean()):5.2f} m/s")
    print(f"    Vent jul     : {float(ws_jul.mean()):5.2f} m/s")

ds.close()