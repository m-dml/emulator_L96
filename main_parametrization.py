from L96_emulator.run_parametrization import run_exp_parametrization, setup_parametrization

if __name__ == '__main__':
    args = setup_parametrization()
    args.pop('conf_exp')
    print(args)
    run_exp_parametrization(**args)