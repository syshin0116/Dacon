public class test{
    public static void main(String[] args){
        // 세로줄
        System.out.println("세로줄:");

        for(int i = 1; i <8; i++){
            int j = 8-i;
            System.out.println(i + "\t" + j + "\t" + i + "\t" + j + "\t" + i + "\t" + j + "\t" + i);
        }

        // 가로줄
        System.out.println("\n가로줄:");
        String row1 = "";
        String row2 = "";
        for (int k = 1; k < 8; k++){
             row1 += k + "\t";
             row2 += 8-k + "\t";
        }
        System.out.println(row1);
        System.out.println(row2);
        System.out.println(row1);
        System.out.println(row2);
        System.out.println(row1);
        System.out.println(row2);
        System.out.println(row1);

    }
}