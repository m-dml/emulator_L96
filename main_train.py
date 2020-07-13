from L96_emulator.run import run_exp, setup

if __name__ == '__main__':
    args = setup()
    args.pop('conf_exp')
    print(args)
    run_exp(**args)