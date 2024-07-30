import numpy as np
from astropy.table import Table
import sys, os, argparse



def main(args):
    
    # tb_org = Table.read('./Input/500pc_refined_memberlist_sub/{}.csv'.format(args.filename))
    tb_org = Table.read('./Input/{}'.format(args.filename))

    rvuseN = 'dr2_radial_velocity'
    rvuseNerr = 'dr2_radial_velocity_error'
    # rvuseN = 'RV_Jackson'
    # rvuseNerr = 'e_RV_Jackson'
    tb_with_rv = tb_org[ (tb_org[rvuseN]>-9000) ]
    
    rv = np.array(tb_with_rv[rvuseN])
    erv   = np.array(tb_with_rv[rvuseNerr])
    
    mass     = tb_org['Mass']
    pmra     = np.array(tb_org['pmra'])
    epmra    = np.array(tb_org['pmra_error'])
    pmdec    = np.array(tb_org['pmdec'])
    epmdec   = np.array(tb_org['pmdec_error'])
    
    
    print("""\n
Target OC name (csv file name): {}
There are in total {} members {}
Only {} ({:1.1%}) of them have 
-rv [{:1.1f},{:1.1f},{:1.1f},{:1.1f}] -fbin 0.5 -pmra [{:1.1f},{:1.1f},{:1.1f},{:1.1f}] -pmdec [{:1.1f},{:1.1f},{:1.1f},{:1.1f}] 
No need to change -walker -steps -burnin for the first run
If run one result is not good, try -walker 50 -steps 35000 -burnin 30000
(only suggestion...)
    """.format(args.filename, len(tb_org), rvuseN, len(tb_with_rv), len(tb_with_rv)/len(tb_org),
               np.mean(rv),    np.std(rv),    np.mean(rv),    np.std(rv),
               np.mean(pmra),  np.std(pmra),  np.mean(pmra),  np.std(pmra),
               np.mean(pmdec), np.std(pmdec), np.mean(pmdec), np.std(pmdec),
               )
    )
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                                     prog        = 'modified the velbin code',
                                     description = '''
                                     modified velbin code.
                                     ''',
                                     epilog = "Contact author: sytang@lowell.edu")
    parser.add_argument("filename",                          action="store",
                        help="Enter your filename you wish you use under ./Input/", type=str)

    # global args
    args = parser.parse_args()
    main(args)