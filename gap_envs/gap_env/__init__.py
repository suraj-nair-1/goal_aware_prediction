from gym.envs.registration import register

register(
    id='GapDoor-v0',
    entry_point='gap_env.door_env:SawyerDoorEnv',
)

register(
    id='GapBlock-v0',
    entry_point='gap_env.block_env:SawyerBlockEnv',
)
