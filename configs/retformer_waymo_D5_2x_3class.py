_base_ = [
         "./retformer_waymo_D1_2x_3class.py"
]

# runner = dict(type='EpochBasedRunner', max_epochs=12)
# evaluation = dict(interval=12)

data = dict(
    train=dict(
        dataset=dict(load_interval=5)
    ),
)
