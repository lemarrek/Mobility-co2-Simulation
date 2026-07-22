import cdsapi

c = cdsapi.Client()

# Hanoi = UTC+7
# 08:00 locale = 01:00 UTC
# 14:00 locale = 07:00 UTC
# 22:00 locale = 15:00 UTC

c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'variable': [
            'boundary_layer_height',
            '10m_u_component_of_wind',
            '10m_v_component_of_wind',
            '2m_temperature',
        ],
        'year': '2023',
        'month': ['01', '07'],
        'day': [f'{d:02d}' for d in range(1, 32)],
        'time': ['01:00', '07:00', '15:00'],  # UTC → locale 08h, 14h, 22h
        'area': [21.5, 105.5, 20.5, 106.5],
        'format': 'netcdf',
    },
    'data/processed/era5_hanoi_v2.nc'
)
print("Téléchargement terminé : era5_hanoi_v2.nc")