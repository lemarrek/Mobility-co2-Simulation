import cdsapi

c = cdsapi.Client()

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
        'month': ['01', '07'],   # janvier (saison sèche) + juillet (mousson)
        'day': [f'{d:02d}' for d in range(1, 32)],
        'time': ['08:00', '14:00', '22:00'],  # tes 3 scénarios
        'area': [21.5, 105.5, 20.5, 106.5],  # bounding box Hanoi [N, W, S, E]
        'format': 'netcdf',
    },
    'era5_hanoi.nc'
)

print("Téléchargement terminé : era5_hanoi.nc")