from L96_emulator.run_DA import run_exp_4DVar, setup_4DVar

if __name__ == '__main__':
    args = setup_4DVar()
    args.pop('conf_exp')
    print(args)
    run_exp_4DVar(**args)