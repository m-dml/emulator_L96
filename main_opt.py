from L96_emulator.run_DA import run_exp_DA, setup_DA

if __name__ == '__main__':
    args = setup_DA()
    args.pop('conf_exp')
    print(args)
    run_exp_DA(**args)